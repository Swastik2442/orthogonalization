import numpy as np

# H = I - (2VV^T)/(V^TV)
# V = X + sign(x1)||X||e1
def computeHouseHolderMatrix(mat: np.ndarray):
    x = mat[:, 0].reshape(-1, 1)
    v = np.copy(x)
    v[0, 0] += np.sign(x[0, 0]) * np.linalg.norm(x)
    v = v / np.linalg.norm(v)
    h = np.identity(mat.shape[0]) - 2 * (v @ v.T)
    return h

def orthogonalisation(mat: np.ndarray):
    hhs = []
    cols = mat.shape[0]
    for i in range(min(mat.shape)):
        hcap = computeHouseHolderMatrix(mat[i:, i:])
        hi = np.identity(cols)
        hi[i:, i:] = hcap
        mat = hi @ mat
        hhs.append(hi)

    q = np.identity(cols)
    for h in hhs[::-1]:
        q = h @ q

    return q, mat

# m = int(input("Enter no. of Rows (m) of the Matrix: "))
# n = int(input("Enter no. of Columns (n) of the Matrix: "))

# matrix = np.array([list(map(float, input(f"Enter the Row {i + 1} (space-separated): ").split())) for i in range(m)])

q, r = orthogonalisation(np.array([[1, -4],[2, 3],[2, 2]]))
print(q, r, q @ r, sep="\n\n")
