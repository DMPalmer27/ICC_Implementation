import galois
from config import SystemContext
from client import DataOwner
from server import Administrator
from utils import generate_vandermonde_G


def main():
    context = SystemContext(q=31, n=10, m=3, d=2, r=1)
    GF = galois.GF(context.q)

    client = DataOwner(context, GF)
    server = Administrator(context, GF)

    # Generate random data of length n over the finite field of size q
    x = GF.Random(context.n)
    # Generate generator matrix (Vandermonde - RS code to achieve equality on m>=r bound because mds)
    G = generate_vandermonde_G(GF, context.m, context.n)


    # Storage Phase
    print("--- Storage Phase ---")
    x_tilde = client.encode_data(x, G)
    print(f"Original x: {x}")
    print(f"Client Secret Key (k): {client.k}")
    print(f"Uploaded x_tilde: {x_tilde}")

    server.store_data(x_tilde, G)

    # Computation Phase
    print("\n--- Computation Phase ---")



    def target_polynomial(data):
        x1, x2, x3 = data[0], data[1], data[2]
        return (x1 ** 2) + (x2 * x3)

    points, results = server.compute_request(target_polynomial)
    print(f"Server evaluated at points: {points}")
    print(f"Server obtained results: {results}")

    # Decoding Phase
    print("\n--- Decoding Phase ---")
    final_answer = client.decode_result(points, results)

    expected = target_polynomial(x)
    print(f"Client decoded answer: {final_answer}")
    print(f"Expected answer f(x): {expected}")

    if final_answer == expected:
        print("\nSUCCESS: The private computation matches the direct computation!")


if __name__ == "__main__":
    main()