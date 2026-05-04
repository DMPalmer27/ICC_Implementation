"""
File: worker.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains general the class for the worker node within the distributed system that performs
polynomial computation on its share of the encoded data
"""

from typing import Callable, Type
import galois

class Worker:

    def __init__(self, GF: Type[galois.FieldArray], data: galois.FieldArray):
        """
        Initializes a worker with data

        :param GF: Galois field object for the scheme
        :param data: Data that this worker holds
        """
        self.GF = GF
        self.data = data

    def evaluate(self, f: Callable):
        """
        Function to evaluate a polynomial at the worker's share of data

        :param f: Polynomial that is being evaluated
        :return: Polynomial's evaluation at the workers share of data
        """
        return f(self.data)