import numpy as np
from config import SystemContext
from utils import get_rm_combinations


class DataOwner:

    def __init__(self, context: SystemContext, GF):
        self.context = context
        self.GF = GF
        self.k = None

    def encode_data(self, x, G):
        self.k = self.GF.Random(self.context.m)
        kG = np.matmul(self.k, G)
        x_tilde = x + kG
        return x_tilde

    def _evaluate_monomials(self, point, exponents):
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
        exponents = get_rm_combinations(self.context.q, self.context.m, self.context.d)

        M_arr = []
        for p in points:
            M_arr.append(self._evaluate_monomials(p, exponents))

        M = self.GF(M_arr)
        res_vec = self.GF(results)

        # Solve for coefficients c
        c = np.linalg.solve(M, res_vec)

        # Evaluate g(k) by plugging our secret key into the solved polynomial
        k_evals = self._evaluate_monomials(self.k, exponents)
        k_evals_gf = self.GF(k_evals)

        # Dot product of coefficients and evaluated terms
        final_answer = np.sum(c * k_evals_gf)
        return final_answer