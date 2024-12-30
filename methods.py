import numpy as np
from scipy.linalg import lu, solve

def power_method(A, v0, tol=1e-6, max_iterations=1000):
    """
    Perform the power method to find the dominant eigenvalue and eigenvector of matrix A.

    Parameters:
        A (numpy.ndarray): The matrix for which the eigenvalue and eigenvector are to be computed.
        v0 (numpy.ndarray): The initial guess for the eigenvector.
        tol (float): The tolerance for convergence (default is 1e-6).
        max_iterations (int): The maximum number of iterations (default is 1000).

    Returns:
        tuple: The dominant eigenvalue and the corresponding eigenvector.
    """
    v = v0 / np.linalg.norm(v0)
    lambda_old = 0

    for _ in range(max_iterations):
        w = np.dot(A, v)

        v = w / np.linalg.norm(w)

        lambda_new = np.dot(v, np.dot(A, v))

        error = abs(lambda_new - lambda_old)

        if error < tol:
            return lambda_new, v

        lambda_old = lambda_new

    raise ValueError("Power method did not converge within the maximum number of iterations.")


def jacobi_method(A, tol=1e-6, max_iterations=1000):
    """
    Perform the Jacobi method to compute the eigenvalues and eigenvectors of a symmetric matrix.

    Parameters:
        A (numpy.ndarray): The symmetric matrix.
        tol (float): Convergence tolerance for the off-diagonal elements (default is 1e-6).
        max_iterations (int): Maximum number of iterations (default is 1000).

    Returns:
        tuple: Eigenvalues (diagonal of A) and eigenvectors (columns of V).
    """
    n = A.shape[0]
    V = np.eye(n)

    for _ in range(max_iterations):
        largest = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > abs(A[p, q]):
                    p, q = i, j
        if abs(A[p, q]) < tol:
            return np.diag(A), V

        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])

        cos = np.cos(theta)
        sin = np.sin(theta)

        G = np.eye(n)
        G[p, p] = cos
        G[q, q] = cos
        G[p, q] = -sin
        G[q, p] = sin

        A = G.T @ A @ G
        V = V @ G

    raise ValueError("Jacobi method did not converge within the maximum number of iterations.")


def lu_inverse(A):
    """
    Find the inverse of a matrix using LU decomposition.

    Parameters:
        A (numpy.ndarray): The matrix to invert.

    Returns:
        numpy.ndarray: The inverse of the matrix.
    """
    P, L, U = lu(A)
    n = A.shape[0]
    A_inv = np.zeros_like(A, dtype=float)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = solve(L, np.dot(P.T, e))
        A_inv[:, i] = solve(U, y)

    return A_inv

def iterative_inverse(A, B, tol=1e-6, max_iterations=100):
    """
    Improve the approximate inverse of a matrix using iterative refinement.

    Parameters:
        A (numpy.ndarray): The original matrix.
        B (numpy.ndarray): The initial approximate inverse.
        tol (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Improved inverse of the matrix.
    """
    for _ in range(max_iterations):
        R = np.eye(A.shape[0]) - np.dot(A, B)
        if np.linalg.norm(R, ord='fro') < tol:
            break
        B += np.dot(B, R)
    return B


def line():
    print('='*15)