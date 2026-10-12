"""
File: client.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the class for the client who has data that is being encoded and updated.
"""

import galois
import numpy as np
from config import SystemContext
from query import DecodedResult, QueryResult, QueryStatus
from quantize import lift_balanced
from utils import build_monomial_matrix, get_information_set
from typing import List, Optional, Type, Union


class Client:

    def __init__(self, context: SystemContext, GF: Type[galois.FieldArray],
                 rng: Optional[Union[int, np.random.Generator]] = None):
        """
        Initializes the client with system parameters

        :param context: SystemContext that holds the scheme parameters
        :param GF: Galois field object for the scheme
        :param rng: Seed or Generator the key is drawn from. Leave as None outside experiments:
            the privacy guarantee assumes k ~ Unif(F_q^m), and a seeded key is a known key.
            np.random.seed does NOT reach galois, which is why this is explicit.
        """
        self.context = context
        self.GF = GF
        self.rng = np.random.default_rng(rng)
        self.k = None
        # Cache of interpolation matrices keyed by (q, m, d, hash of the evaluation points).
        # M depends only on the information set and the monomial exponents, so it is identical
        # across every gradient component and every gradient-descent step of a training run.
        self._M_cache = {}

    def encode_data(self, x: galois.FieldArray, G: galois.FieldArray) -> galois.FieldArray:
        """
        This function performs data smoothing and encoding by the client

        If the context declares max_input, every stored value must read as an integer code with
        |x_i| <= max_input. This is the assumption the server's overflow check rests on, and
        upload is the last moment it can be checked: afterwards the client keeps only k, and
        the provider only ever sees x masked by kG.

        :param x: Data being encoded
        :param G: mxn matrix which is performing smoothing and encoding
        :return: Matrix containing data that has been smoothed and encoded
        :raises ValueError: if a stored code exceeds the declared max_input
        """
        if self.context.max_input is not None:
            worst = int(np.max(np.abs(lift_balanced(x, self.context.q)))) if x.size else 0
            if worst > self.context.max_input:
                raise ValueError(
                    f"data has |x_i| = {worst} > declared max_input = "
                    f"{self.context.max_input}; the server's overflow check would be unsound. "
                    f"Quantise with quantize.to_integer, which clips to the declared range."
                )
        self.k = self.GF.Random(self.context.m, seed=self.rng)
        kG = np.matmul(self.k, G)
        x_tilde = x + kG
        return x_tilde

    def _interpolation_matrix(self, points, exponents) -> galois.FieldArray:
        """
        Returns the interpolation matrix for the given points, building it on first use.

        Keyed by (q, m, d) plus a hash of the evaluation points, so a caller that passes a
        different point set still gets a correct matrix rather than a stale cache hit.

        :param points: Evaluation points defining the linear system
        :param exponents: Monomial exponents of the information set
        :return: GF array of shape (len(points), len(exponents))
        """
        P = points if isinstance(points, self.GF) else self.GF(np.asarray(points))
        key = (self.GF.order, self.context.m, self.context.d,
               hash(np.asarray(P).tobytes()))

        if key not in self._M_cache:
            self._M_cache[key] = build_monomial_matrix(self.GF, P, exponents)
        return self._M_cache[key]

    def decode_result(self, result: QueryResult) -> DecodedResult:
        """
        Decodes one server response. A refused query passes its status and reason straight
        through, so the caller always learns whether the value can be trusted.

        :param result: QueryResult from Server.compute_request
        :return: DecodedResult holding g(k) = f(x) if the query was accepted, else the refusal
        """
        return self.decode_results([result])[0]

    def decode_results(self, results: List[QueryResult]) -> List[DecodedResult]:
        """
        Decodes several server responses with one Gaussian elimination, which is how a
        gradient with P components is recovered in a single pass. Refused queries are passed
        through with their status; only the accepted ones are solved.

        :param results: QueryResults from Server.compute_request, all from the same server
        :return: One DecodedResult per input, in order
        """
        accepted = [i for i, r in enumerate(results) if r.ok]
        out = [DecodedResult(r.status, reason=r.reason) for r in results]
        if not accepted:
            return out

        points = results[accepted[0]].points
        batch = self.GF(np.stack([np.asarray(self.GF(results[i].results)) for i in accepted],
                                 axis=1))
        values = self._interpolate_at_key(points, batch)
        for col, i in enumerate(accepted):
            out[i] = DecodedResult(QueryStatus.OK, value=values[col])
        return out

    def _interpolate_at_key(self, points, results):
        """
        Decodes the final result by reconstructing the polynomial over the encoded data and evaluating it
        at the stored secret key.

        Accepts either a single computation or a batch. Passing results of shape (lambda, P)
        decodes P polynomials in one Gaussian elimination.

        :param points: Points that the monomials are evaluated at in order to get the system we are solving
        :param results: Results from server, shape (lambda,) for one polynomial or (lambda, P) for a batch
        :return: The recovered polynomial's value at the key k; a scalar, or a length-P vector for a batch
        """
        exponents = get_information_set(self.context.q, self.context.m, self.context.d)

        M = self._interpolation_matrix(points, exponents)
        res_vec = results if isinstance(results, self.GF) else self.GF(results)

        # Solve for coefficients. np.linalg.solve handles a matrix right-hand side, so a
        # batch of P polynomials costs one elimination rather than P of them.
        c = np.linalg.solve(M, res_vec)

        # Evaluate g(k) by plugging our secret key into the solved polynomial
        k_evals = build_monomial_matrix(self.GF, self.k, exponents)[0]

        if c.ndim == 1:
            return np.sum(c * k_evals)
        # Batched: contract the monomial evaluations against each column of coefficients
        return k_evals @ c