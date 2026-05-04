"""
File: main.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file contains the script which actually runs and tests an instance of the scheme
"""

import galois
import numpy as np
from config import SystemContext
from client import Client
from server import Server
from utils import generate_vandermonde_G, compute_p_entropy, compute_max_subset_p_entropy, compute_required_m, \
    compute_leakage_bound, generate_random_G


def main():

    # Initialize system parameters (using default values given in utils.py)
    context = SystemContext() # Not super low because my computer cannot tolerate
    GF = galois.GF(context.q)

    # Simulate non-uniform data
    # Generate weighted probability distribution
    num_weighted = 5
    probs = np.zeros(context.q)
    probs[:num_weighted] = 0.9 / num_weighted # Give first num_weighted values 90% of the mass
    probs[num_weighted:] = 0.1 / (context.q - num_weighted) # Give remaining values remaining 10% of the mass

    rng = np.random.default_rng()
    samples = rng.choice(context.q, size=context.n, p=probs)
    x = GF(samples)


    # Compute entropy quantities over the data and derive m (key size) based off of them and store in context
    context.H_p_X = compute_p_entropy(x, context.q, context.p)
    context.max_H_p_X_R = compute_max_subset_p_entropy(x, context.q, context.p, context.r)
    context.m = compute_required_m(context)
    eps_c = compute_leakage_bound(context)


    print(f"Data distribution: biased (elements 0-{num_weighted-1} have 90% mass)")
    print(f"H_{context.p}(X)              = {context.H_p_X:.4f}  (p-entropy of full data)")
    print(f"max_R H_{context.p}(X_R)      = {context.max_H_p_X_R:.4f}  (worst r-subset entropy)")
    print(f"Theorem 1 lower bound on m = {context.m}")
    print(f"Theoretical leakage ε_c    = {eps_c:.6f}  (per r-subset)")
    print(f"Key size m set to          = {context.m}")

    client = Client(context, GF)
    server = Server(context, GF)

    G = generate_random_G(GF, context.m, context.n)

    # Storage Phase
    print("--- Storage Phase ---")
    x_tilde = client.encode_data(x, G)
    print(f"Original x: {x}")
    print(f"Client Secret Key (k): {client.k}")
    print(f"Uploaded x_tilde: {x_tilde}")
    print(f"Encoded data entropy: {compute_p_entropy(x_tilde, context.q, context.p)}")

    server.store_data(x_tilde, G)

    # Computation Phase
    print("\n--- Computation Phase ---")


    # Random polynomial that I chose to use for this toy example
    def target_polynomial(data):
        x1, x2, x3 = data[0], data[1], data[2]
        return (x1 ** 2) + (x2 * x3)

    points, results = server.compute_request(target_polynomial)
    # print(f"Server evaluated at points (RM information set): {points}")
    # print(f"Server obtained results: {results}")

    # Decoding Phase
    print("\n--- Decoding Phase ---")
    final_answer = client.decode_result(points, results)

    expected = target_polynomial(x)
    print(f"Client decoded answer: {final_answer}")
    print(f"Expected answer f(x): {expected}")

    if final_answer == expected:
        print("\nSUCCESS: The private computation matches the direct computation!")


    # Overall Privacy Analysis
    print("\n--- ICC Privacy Guarantee Summary ---")
    print(f"Leakage I(X̃; X_R) <= ε_c = {eps_c:.6f}  for every r={context.r}-subset R")
    print(f"This bound holds with probability >= 1 - 1/a = {1 - 1/context.a:.2f}")
    print(f"over the random choice of G  (ICC paper Theorem 1)")


if __name__ == "__main__":
    main()