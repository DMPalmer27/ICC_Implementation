"""
File: client.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the class for the client who has data that is being encoded and updated.
"""

import galois
import numpy as np
from config import SystemContext
from utils import get_information_set
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

    def _evaluate_monomials(self, point, exponents) -> List:
        """
        Evaluates monomials defined by exponents at a given field element point

        :param point: Given point monomials are being evaluated at
        :param exponents: Exponents for the monomial evaluations
        :return: List of monomial evaluations at the point
        """
        vals = []
        for exp in exponents:
            # Calculate (point[0]^exp[0]) * (point[1]^exp[1])...
            term_val = self.GF(1)
            for i in range(self.context.m):
                if exp[i] > 0:
                    term_val *= point[i] ** exp[i]
            vals.append(term_val)
        return vals

    def decode_result(self, points, results):
        """
        Decodes the final result by reconstructing the polynomial over the encoded data and evaluating it
        at the stored secret key.

        :param points: Points that the monomials are evaluated at in order to get the system we are solving
        :param results: Results from server that are being decoded
        :return: The recovered polynomials value at the key k
        """
        exponents = get_information_set(self.context.q, self.context.m, self.context.d)

        M_arr = []
        for p in points:
            M_arr.append(self._evaluate_monomials(p, exponents))

        M = self.GF(M_arr)
        res_vec = self.GF(results)

        # Solve for coefficients
        c = np.linalg.solve(M, res_vec)

        # Evaluate g(k) by plugging our secret key into the solved polynomial
        k_evals = self._evaluate_monomials(self.k, exponents)
        k_evals_gf = self.GF(k_evals)

        final_answer = np.sum(c * k_evals_gf)
        return final_answer