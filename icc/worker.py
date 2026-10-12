"""
File: worker.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains general the class for the worker node within the distributed system that performs
polynomial computation on its share of the encoded data
"""

import galois
from query import Polynomial

class Worker:

    def __init__(self, data: galois.FieldArray):
        """
        Initializes a worker with data

        :param data: Data that this worker holds
        """
        self.data = data

    def evaluate(self, f: Polynomial):
        """
        Function to evaluate a polynomial at the worker's share of data

        :param f: Polynomial that is being evaluated; already checked by the server
        :return: Polynomial's evaluation at the workers share of data
        """
        return f(self.data)