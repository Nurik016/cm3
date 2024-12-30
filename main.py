import numpy as np
from methods import *

# 2
A = np.array([[50, 107, 36],
              [35, 54, 20],
              [31, 66, 21]])
A_inv = lu_inverse(A)
print("Inverse of A using LU factorization:")
print(A_inv)
line()


# 3
A = np.array([[1, 10, 1],
              [2, 0, 1],
              [3, 3, 2]])
B = np.array([[0.4, 2.4, -1.4],
              [0.14, 0.14, -0.14],
              [-0.85, -3.8, 2.8]])
improved_B = iterative_inverse(A, B)
print("Improved inverse of A using the iterative method:")
print(improved_B)
line()


# 4
# Define the matrix and initial vector
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])
v0 = np.array([1, 0, 0])
eigenvalue, eigenvector = power_method(A, v0)
print("Largest Eigenvalue:", eigenvalue)
print("Corresponding Eigenvector:", eigenvector)
line()


# 5
# Define the matrix
A = np.array([[1, np.sqrt(2), 2],
              [np.sqrt(2), 3, np.sqrt(2)],
              [2, np.sqrt(2), 1]])
eigenvalues, eigenvectors = jacobi_method(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

