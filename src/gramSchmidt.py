import numpy as np

def qr_by_gram_schmidt(mat: np.ndarray):
    m, n = mat.shape
    q = np.zeros((m, m))
    r = np.zeros((n, n))

    for j in range(n):
        v = mat[:, j]
        for i in range(j):
            r[i, j] = q[:, i].T @ mat[:, j]
            v = v.squeeze() - (r[i, j] * q[:, i])
        r[j, j] = np.linalg.norm(v)
        q[:, j] = (v / r[j, j]).squeeze()
    
    return q, r

ans = qr_by_gram_schmidt(np.array([[1, 1, 0], [0, 0, 3], [1, -1, 4]]))

print("Q : ", ans[0])
print("R : ", ans[1])
print("Q.R : ", ans[0] @ ans[1])
