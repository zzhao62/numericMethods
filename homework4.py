import numpy as np
import matplotlib.pyplot as plt
"""
@Author Zhuo Zhao
@Date 11/8/2021
"""
def exact_solution(m, dx, T, L=2*np.pi):
    """
    :param m:
    :param dx:
    :param T:
    :param L:
    :return:
    """
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)

    u = np.exp(-np.power(m,2)*T)*np.sin(m*x)
    return x, u

def init_eq(m, dx, r, T, L):
    I = lambda x: np.sin(m * x)
    Nx = int(round(L / dx))
    dt = r * np.power(dx, 2)
    Nt = int(round(T / float(dt)))
    x = np.linspace(0, L, Nx + 1)

    u = np.zeros(Nx + 1)  # The final solution Array
    u_n = np.zeros(Nx + 1)  # The previous time array
    return I, Nx, Nt, x, u, u_n

def TDMASolver(lower, mid, upper, b):
    """
    Solve Tri-diaganol matrix
    """
    n = len(b)
    lowercp, midcp, uppercp, bcp = map(lambda x: np.array(x, np.float32), (lower, mid, upper, b))
    for i in range(1, n):
        sc = lowercp[i-1]/midcp[i-1]
        midcp[i] = midcp[i] - sc*uppercp[i-1]
        bcp[i] = bcp[i] - sc*bcp[i-1]

    xc = midcp
    xc[-1] = bcp[-1]/midcp[-1]

    for m in range(n-2, -1, -1):
        xc[m] = (bcp[m]-uppercp[m]*xc[m+1])/midcp[m]

    return xc

def test_TDMA():
    A = np.array([[2, 3, 0, 0], [3, 1, 4, 0], [0, 1, 2, 9], [0, 0, 1, 4]], dtype=float)

    lower = np.array([3., 1, 1])
    mid = np.array([2., 1., 2., 4.])
    upper = np.array([3., 4., 9.])
    b = np.array([3, 4, 5, 6.])
    print("TDMA Result", TDMASolver(lower, mid, upper, b))
    print("Numpy Result", np.linalg.solve(A,b))


def FE_method(m, dx, r, T, L=2*np.pi):
    """
    :param m: parameter m
    :param L: X range
    :param dx: x step
    :param r: dt/dx^2
    :param T: Time range
    :return:
    """
    I, Nx, Nt, x, u, u_n = init_eq(m, dx, r, T, L)

    # Set up Initial Condition
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    # The Foward Euler Method
    for i in range(0, Nt):
        u[1:Nx] = u_n[1:Nx] + r*(u_n[0:Nx-1]-2*u_n[1:Nx]+u_n[2:Nx+1])
        # Boundary Conditions
        u[0] = 0
        u[Nx] = 0
        u_n, u = u, u_n

    return x, u

def BE_TDMA(m, dx, r, T, L=2*np.pi):
    """
    :param m: parameter m
    :param L: X range
    :param dx: x step
    :param r: dt/dx^2
    :param T: Time range
    :return:
    """
    I, Nx, Nt, x, u, u_n = init_eq(m, dx, r, T, L)

    # Set up Initial Condition
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    b = np.zeros(Nx+1)

    lower = np.zeros(Nx)
    mid = np.zeros(Nx+1)
    upper = np.zeros(Nx)
    lower[:] = -r
    mid[:] = 1+2*r
    upper[:] = -r

    for n in range(0, Nt):
        for i in range(1, Nx):
            b[i] = u_n[i]
        b[0] = b[Nx] = 0
        u[:] = TDMASolver(lower, mid, upper, b)
        # Update u_n before next step
        u_n[:] = u
    return x, u

def BE_method(m, dx, r, T, L=2*np.pi):
    """
    :param m: parameter m
    :param L: X range
    :param dx: x step
    :param r: dt/dx^2
    :param T: Time range
    :return:
    """
    I, Nx, Nt, x, u, u_n = init_eq(m, dx, r, T, L)

    # Set up Initial Condition
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    A = np.zeros((Nx+1, Nx+1))
    b = np.zeros(Nx+1)

    for i in range(1, Nx):
        A[i,i-1] = -r
        A[i,i+1] = -r
        A[i,i] = 1 + 2*r

    A[0,0] = 1+2*r
    A[Nx, Nx] = 1+2*r
    import scipy.linalg

    for n in range(0, Nt):
        # Compute b and solve linear system
        for i in range(1, Nx):
            b[i] = u_n[i]
        b[0] = b[Nx] = 0
        u[:] = scipy.linalg.solve(A, b)

        # Update u_n before next step
        u_n[:] = u
    return x, u

def CN_TDMA(m, dx, r, T, L=2*np.pi):
    """
    :param m: parameter m
    :param L: X range
    :param dx: x step
    :param r: dt/dx^2
    :param T: Time range
    :return:
    """
    I, Nx, Nt, x, u, u_n = init_eq(m, dx, r, T, L)

    # Set up Initial Condition
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    b = np.zeros(Nx+1)

    lower = np.zeros(Nx)
    mid = np.zeros(Nx+1)
    upper = np.zeros(Nx)
    lower[:] = -0.5*r
    mid[:] = 1+r
    upper[:] = -0.5*r

    for n in range(0, Nt):
        # Compute b and solve linear system
        for i in range(1, Nx):
            b[i] = u_n[i] + 0.5*r*(u_n[i-1] - 2 * u_n[i] + u_n[i+1])
        b[0] = b[Nx] = 0
        u[:] = TDMASolver(lower, mid, upper, b)
        # Update u_n before next step
        u_n[:] = u
    return x, u

def CN_method(m, dx, r, T, L=2*np.pi):
    """
    :param m: parameter m
    :param L: X range
    :param dx: x step
    :param r: dt/dx^2
    :param T: Time range
    :return:
    """
    I, Nx, Nt, x, u, u_n = init_eq(m, dx, r, T, L)

    # Set up Initial Condition
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    A = np.zeros((Nx+1, Nx+1))
    B = np.zeros((Nx+1, Nx+1))
    b = np.zeros(Nx+1)

    for i in range(1, Nx):
        A[i,i-1] = -0.5*r ; B[i,i-1] = 0.5*r
        A[i,i+1] = -0.5*r ; B[i,i+1] = 0.5*r
        A[i,i] = 1 + r ; B[i,i] = 1 - r

    A[0,0] = 1+r ; B[0,0] = 1-r
    A[Nx, Nx] = 1+r ; B[Nx, Nx] = 1-r
    C = np.matmul(np.linalg.inv(B), A)
    import scipy.linalg

    for n in range(0, Nt):
        # Compute b and solve linear system
        for i in range(1, Nx):
            b[i] = u_n[i]
        b[0] = b[Nx] = 0
        u[:] = scipy.linalg.solve(C, b)

        # Update u_n before next step
        u_n[:] = u
    return x, u

def plot_result(m, dx, r, T):
    x, u_exact = exact_solution(m=m, dx=dx, r=r, T=T)
    xf, uf = FE_method(m=m, dx=dx, r=r, T=T)
    xbTDMA, ubTDMA = BE_TDMA(m=m, dx=dx, r=r, T=T)
    cTDMA, ucTDMA = CN_TDMA(m=m, dx=dx, r=r, T=T)



if __name__ == '__main__':
    xc, uc = CN_method(2, dx=2*np.pi/20, r=1/3, T=1)
    xcTDMA, ucTDMA = CN_TDMA(2, dx=2*np.pi/20, r=1/3, T=1)
    x, u_exact = exact_solution(2, dx=2*np.pi/20, T=1)
    #print("Time 1 FE", uf)
    #print("Time 1 BE", ub)
    #print("Time 1 BETDMA", ubTDMA)
    print("Time 1 CN", uc)
    print("Time 1 CNTDMA", ucTDMA)
    #print(u_exact)
    plt.ylim()
    #plt.plot(xf, uf, label="FE Method")
    #plt.plot(x, u_exact, label="Exact Solution")
    #plt.plot(xb, ub, label="BE Method")
    #plt.plot(xbTDMA, ubTDMA, label="BE TDMA")
    plt.plot(xc, uc, label="CN Method")
    plt.plot(xcTDMA, ucTDMA, label="CN TDMA")
    plt.plot(x, u_exact, label="Exact")
    plt.legend()
    plt.show()
    test_TDMA()
