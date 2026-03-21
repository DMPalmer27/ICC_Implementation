import itertools

def generate_vandermonde_G(GF, m, n):
    """
    Generates an m x n Vandermonde matrix over the provided Galois Field.
    This guarantees the MDS property, achieving the optimal m = r bound.
    """
    # 1. Enforce the mathematical rule for distinct columns
    if GF.order < n + 1:
        raise ValueError(
            f"Field size q={GF.order} is too small! Must be >= {n + 1} to create an MDS matrix of length n.")

    # 2. Select n distinct, non-zero elements from the field
    # (GF.elements returns all elements [0, 1, 2...]. We slice from 1 to avoid 0)
    alphas = GF.elements[1:n + 1]

    # 3. Construct the matrix
    G = GF.Zeros((m, n))
    for i in range(m):
        for j in range(n):
            # In polynomial evaluation, the data 'x' is the constant term (power 0).
            # The key 'k' acts as the higher coefficients (power 1 to m).
            # So, row i corresponds to the power (i + 1).
            G[i, j] = alphas[j] ** (i + 1)

    return G


def get_rm_combinations(q: int, m: int, d: int):
    """
    Generates the core combinatorial tuples for Reed-Muller RM_q(d, m).
    Used as evaluation points by the Server and as polynomial exponents by the Client.
    """
    combos = []
    for combo in itertools.product(range(q), repeat=m):
        if sum(combo) <= d:
            combos.append(combo)
    return combos