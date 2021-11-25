import numpy as np

def jacobi_solver(A, b, N=100):
    """
    Use Jacobi Method to solve the linear system
    """
    x = np.zeros(A.shape[0])
    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(N):
        x = (b - np.dot(R,x)) / D

    return x

if __name__ == '__main__':
    A = np.array([[2.0, 1.0], [5.0, 7.0]])
    b = np.array([11.0, 13.0])
    sol = jacobi_solver(A,b)
    print(sol)