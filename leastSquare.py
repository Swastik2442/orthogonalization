import numpy as np

def leastSquare(matrix, b):
    aTrasnposeA = (np.transpose(matrix) @ matrix)
    inv = np.linalg.inv(aTrasnposeA)
    invAtranspose = (inv @ np.transpose(matrix))
    x = (invAtranspose @ b)
    return x

a = np.array([[1, 2], 
         [2, 3],
         [3, 4],
         [4, 5]])

b = np.array([[5], [7], [9], [12]])

ansX = leastSquare(a, b)

print("Ouput Value X : ", ansX)