import numpy as np

def getGivens(n: int, m: int, i: int, k: int, theta: float = None, x: np.ndarray = None) -> np.ndarray:
    assert i != k, "i and k must be different"
    assert n > 0 and m > 0, "n and m must be greater than 0"
    assert i < n and i < m and k < n and k < m, "i and k must be less than n and m"
    assert not (theta is None and x is None), "Either theta or x must be provided"

    if theta is None:
        xi, xk = x[i], x[k]
        deno = np.sqrt(xi**2 + xk**2)
        cos = xi / deno
        sin = -xk / deno
    else:
        cos = np.cos(theta)
        sin = np.sin(theta)

    mat = np.eye(n, m)
    mat[i][i] = cos
    mat[k][k] = cos
    mat[i][k] = sin
    mat[k][i] = -sin

    return mat

def getQR(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = a.shape    
    q = np.eye(n, m)
    lowerIndices = ((i, j) for i in range(1, n) for j in range(i) if np.absolute(a[i][j]) > np.finfo(float).eps)
    for i, j in lowerIndices:
        g = getGivens(n, m, j, i, x=a.T[j])
        a = g.T @ a
        q = q @ g

    return q, a

if __name__ == "__main__":
    a = np.array([[2, -1, -2], [-4, 6, 3], [-4, -2, 8]])
    q, r = getQR(a)
    print(a, q, r, q@r, sep="\n\n")
