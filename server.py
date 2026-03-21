import numpy as np
from typing import Callable, List
from config import SystemContext
from worker import WorkerNode
from utils import get_rm_combinations


class Administrator:
    """Manages the storage and delegates evaluation tasks."""

    def __init__(self, context: SystemContext, GF):
        self.context = context
        self.GF = GF
        self.x_tilde = None
        self.G = None

    def store_data(self, x_tilde, G):
        self.x_tilde = x_tilde
        self.G = G

    def _get_evaluation_points(self) -> List:
        """Dynamically returns the RM_q(d, m) info set for any dimension."""
        base_points = get_rm_combinations(self.context.q, self.context.m, self.context.d)
        # Wrap in GF because server requires finite field arithmetic
        return [self.GF(list(combo)) for combo in base_points]

    def compute_request(self, f: Callable):
        points = self._get_evaluation_points()
        results = []

        workers = [WorkerNode(self.GF) for _ in range(len(points))]

        for i, t in enumerate(points):
            tG = np.matmul(t, self.G)
            shifted_data = self.x_tilde - tG

            res = workers[i].evaluate(f, shifted_data)
            results.append(res)

        return points, results