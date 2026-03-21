from typing import Callable

class WorkerNode:
    """Represents a computing node performing the actual computation."""
    def __init__(self, GF):
        self.GF = GF

    def evaluate(self, f: Callable, shifted_data):
        """Computes f on the blinded data provided by the Admin."""
        return f(shifted_data)