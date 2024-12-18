import numpy as np

# H = I - (2VV^T)/(V^TV)
# V = X + sign(x1)||X||e1
def computeHouseHolderMatrix(mat: np.ndarray):
    v = mat[:, 0].reshape(-1, 1).copy()
    v[0, 0] += np.sign(v[0, 0]) * np.linalg.norm(v)
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

m = int(input("Enter no. of Rows (m) of the Matrix: "))
n = int(input("Enter no. of Columns (n) of the Matrix: "))

matrix = np.array([list(map(float, input(f"Enter the Row {i + 1} (space-separated): ").split())) for i in range(m)])

q, r = orthogonalisation(matrix)

print("\nQ : \n", q, "\nR : \n", r, "\nQ*R : \n", (q@r))

