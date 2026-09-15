"""
File: client.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the class for the client who has data that is being encoded and updated.
"""

import galois
import numpy as np
from config import SystemContext
from utils import build_monomial_matrix, get_information_set
from typing import Type, List


class Client:

    def __init__(self, context: SystemContext, GF: Type[galois.FieldArray]):
        """
        Initializes the client with system parameters

        :param context: SystemContext that holds the scheme parameters
        :param GF: Galois field object for the scheme
        """
        self.context = context
        self.GF = GF
        self.k = None
        # Cache of interpolation matrices keyed by (q, m, d, hash of the evaluation points).
        # M depends only on the information set and the monomial exponents, so it is identical
        # across every gradient component and every gradient-descent step of a training run.
        self._M_cache = {}

    def encode_data(self, x: galois.FieldArray, G: galois.FieldArray) -> galois.FieldArray:
        """
        This function performs data smoothing and encoding by the client

        :param x: Data being encoded
        :param G: mxn matrix which is performing smoothing and encoding
        :return: Matrix containing data that has been smoothed and encoded
        """
        self.k = self.GF.Random(self.context.m)
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

    def decode_result(self, points, results):
        """
        Decodes the final result by reconstructing the polynomial over the encoded data and evaluating it
        at the stored secret key.

        Accepts either a single computation or a batch. Passing results of shape (lambda, P)
        decodes P polynomials in one Gaussian elimination, which is how a gradient with P
        components is recovered in a single pass.

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