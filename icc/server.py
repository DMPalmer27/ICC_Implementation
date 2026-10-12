"""
File: server.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the admin class which receives encoded data, distributes it to the workers, sends
polynomial to the workers to calculate, and returns aggregated result from workers.
"""

import numpy as np
import galois
from config import SystemContext
from query import Polynomial, QueryResult, QueryStatus, check_query
from worker import Worker
from utils import get_information_set
from typing import Type


class Server:

    def __init__(self, context: SystemContext, GF: Type[galois.FieldArray]):
        """
        Initializes the server with system parameters

        :param context: System parameters for the scheme
        :param GF: Galois field object for the scheme
        """
        self.context = context
        self.GF = GF
        self.x_tilde = None
        self.G = None
        self.workers = []
        self.points = []

    def store_data(self, x_tilde: galois.FieldArray, G: galois.FieldArray):
        """
        This function is what the user calls to give data to the admin. It takes the encoded data, creates each worker,
        and gives each worker the encoded data shifted by the information set evaluation points and the passed in
        generator matrix which allows for successful decoding.

        :param x_tilde: Data that has been encoded by the user
        :param G: Generator matrix that the user used, with the random key, to smooth and encode their data
        """
        self.x_tilde = x_tilde
        self.G = G
        self.points = self._get_evaluation_points()

        # Shift the stored data by tG for every evaluation point t at once. This is one
        # (lambda x m) @ (m x n) product rather than lambda separate (m,) @ (m, n) matmuls
        # in a Python loop; x_tilde broadcasts across the rows.
        T = self.GF(np.asarray([np.asarray(p) for p in self.points]))
        shares = self.x_tilde - (T @ self.G)

        self.workers = [Worker(shares[i]) for i in range(shares.shape[0])]

    def _get_evaluation_points(self) -> list[galois.FieldArray]:
        """
        Gets the evaluation points of the information set for data storage. Wraps in a Galois field because
        the server requires GF arithmetic

        :return: Evaluation points wrapped in GF
        """
        base_points = get_information_set(self.context.q, self.context.m, self.context.d)
        # Wrap in GF because server requires finite field arithmetic
        return [self.GF(list(combo)) for combo in base_points]

    def compute_request(self, f: Polynomial) -> QueryResult:
        """
        This is the function that the user calls when they want to calculate a polynomial over their data.

        The query is checked BEFORE any worker is contacted (query.check_query): a query whose
        degree exceeds d, or whose worst case over the declared input range could leave the
        balanced window of F_q, is refused with a status and reason instead of returning a
        result that may silently be wrong. The check uses only f's coefficients and public
        parameters, so a refusal leaks nothing about the data.

        :param f: The polynomial the user wants computed, with explicit integer coefficients
        :return: QueryResult; when ok, it holds the points the server evaluated at and the results for those
        points, which the user interpolates to recover the polynomial whose value at the secret key is the
        desired computation
        """
        if not isinstance(f, Polynomial):
            raise TypeError("queries must be query.Polynomial objects; an opaque callable "
                            "cannot be checked for overflow or degree")
        if f.num_variables > self.context.n:
            raise ValueError(f"query uses x{f.num_variables - 1} but the data has length "
                             f"{self.context.n}")

        check = check_query(f, self.context.q, self.context.d, self.context.max_input)
        if not check.ok:
            return check

        results = [worker.evaluate(f) for worker in self.workers]
        return QueryResult(QueryStatus.OK, points=self.points, results=results,
                           bound=check.bound, window=check.window)