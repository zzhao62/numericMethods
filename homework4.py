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
    x, u_exact = exact_solution(m=m, dx=dx, T=T)
    xf, uf = FE_method(m=m, dx=dx, r=r, T=T)
    xbTDMA, ubTDMA = BE_TDMA(m=m, dx=dx, r=r, T=T)
    xcTDMA, ucTDMA = CN_TDMA(m=m, dx=dx, r=r, T=T)
    print("FE max difference", max(np.abs(u_exact-uf)))
    print("BE max difference", max(np.abs(u_exact-ubTDMA)))
    print("CN max difference", max(np.abs(u_exact-ucTDMA)))
    plt.figure(figsize=(6,4))
    plt.ylim(-1,1)
    plt.plot(x, u_exact, label="Exact Solution")
    plt.plot(xf, uf, label="Forward Euler")
    plt.plot(xbTDMA, ubTDMA, label="Backward Euler TDMA")
    plt.plot(xcTDMA, ucTDMA, label="Crank Nicolson TDMA")
    name = "T={}, m={}, dx={}, r={}".format(T, m, round(dx,2), round(r,2))
    plt.title(name)
    plt.legend(fontsize=6)
    saveName = ""
    if (m == 2):
        saveName = "c{}".format(T)
    else:
        saveName = "m={}".format(m)

    plt.savefig("{}.png".format(saveName))
    plt.show()

def error_analysis(method, name):
    m = 2
    T = 0.5
    X = 1
    r = 1/3
    n = 20
    error = np.zeros(n-2)
    dxs = np.zeros(n-2)
    for i in range(2,n):
        dx = 2*np.pi/(i*10)
        dxs[i-2] = dx**2
        x, u_e = exact_solution(m=m, dx=dx, T=T)
        xp, u_p = method(m=m, dx=dx, r=r, T=T)
        p = round(X/(2*np.pi)*len(x))
        error[i-2] = np.abs(u_e[p] - u_p[p])
    plt.figure(figsize=(6,4))
    plt.plot(dxs, error)
    plt.title("{} Error Analysis".format(name))
    plt.xlabel("dx^2")
    plt.ylabel("error at X=pi When Time=0.5")
    plt.savefig("{}error.png".format(name[:2]))
    plt.show()
    return error

def stability_anlysis(method, name, r=1/3):
    m = 2
    X = np.pi
    ts = np.linspace(0,2,100)
    dx = 2*np.pi/10
    n = ts.shape[0]
    error = np.zeros(n)
    for i in range(n):
        x, u_e = exact_solution(m=m, dx=dx, T=ts[i])
        xt, u_t = method(m=m, dx=dx, r=r, T=ts[i])
        p = round(X/(2*np.pi)*len(x))
        error[i] = np.abs(u_e[p] - u_t[p])
    plt.figure(figsize=(6,4))
    plt.plot(ts, error)
    plt.title("{} Stability Analysis".format(name))
    plt.xlabel("Time")
    plt.ylabel("Error at X=pi")
    plt.savefig("{}stability.png".format(name[:2]))
    plt.show()
    return error

def amplicationFactors():
    p = np.linspace(0,2,100)
    Ae = np.exp(-2*np.power(p,2))
    Af = 1 - 2*np.power(np.sin(p),2)
    Ab = 1/(1+2*np.power(np.sin(p),2))
    Ac = (1-np.power(np.sin(p),2))/(1+np.power(np.sin(p),2))
    plt.ylim((-1.5,1.5))
    plt.xlabel("p")
    plt.ylabel("A(p)")
    plt.title("r=0.5")
    plt.plot(p,Ae, label="Exact Solution")
    plt.plot(p,Af, label="Forward Euler")
    plt.plot(p,Ab, label="Backward Euler")
    plt.plot(p,Ac, label="Crank Nicolson")
    plt.legend()
    plt.savefig("amplication.png")
    plt.show()

if __name__ == '__main__':
    #lot_result(2, 2*np.pi/20, 1/3, 0.1)
    #plot_result(2, 2*np.pi/20, 1/3, 1)
    #error_analysis(FE_method,"FE Method")
    #error_analysis(BE_TDMA, "BE TDMA")
    #e#rror_analysis(CN_TDMA, "CN TDMA")
    #stability_anlysis(FE_method, "FE Method")
    #stability_anlysis(BE_TDMA, "BE TDMA")
    #stability_anlysis(CN_TDMA, "CN TDMA")
    stability_anlysis(BE_TDMA, "Backward Euler when r=2", r=2)
    stability_anlysis(CN_TDMA, "Crank Nicolson when r=2", r=2)
    #for m in [3,5,7]:
    #    plot_result(m, 2 * np.pi / 20, 0.5, 0.1)
    amplicationFactors()

