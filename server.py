"""
File: server.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the admin class which receives encoded data, distributes it to the workers, sends
polynomial to the workers to calculate, and returns aggregated result from workers.
"""

import numpy as np
import galois
from typing import Callable, List
from config import SystemContext
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
        self.workers = []

        # Create workers and give each worker a piece of the data
        for p in self.points:
            tG = np.matmul(p, self.G)
            shifted_data = self.x_tilde - tG
            self.workers.append(Worker(shifted_data))

    def _get_evaluation_points(self) -> list[galois.FieldArray]:
        """
        Gets the evaluation points of the information set for data storage. Wraps in a Galois field because
        the server requires GF arithmetic

        :return: Evaluation points wrapped in GF
        """
        base_points = get_information_set(self.context.q, self.context.m, self.context.d)
        # Wrap in GF because server requires finite field arithmetic
        return [self.GF(list(combo)) for combo in base_points]

    def compute_request(self, f: Callable) -> tuple[List, List]:
        """
        This is the function that the user calls when they want to calculate a polynomial over their data.

        :param f: The polynomial the user wants computed
        :return: The points that the server evaluated the polynomial at and the results for those points. The user
        uses these results to extract the polynomial that, when evaluated at the secret key, gives their desired
        computation.
        """
        results = [worker.evaluate(f) for worker in self.workers]
        return self.points, results