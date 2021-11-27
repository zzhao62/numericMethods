import numpy as np

def generate_u(guess, dx=np.pi/10, dy=np.pi/10, x_range=2*np.pi, y_range=2*np.pi):
    Nx = int(x_range / dx) + 1
    Ny = int(y_range / dy) + 1
    dxs = np.linspace(0,x_range,Nx)
    dys = np.linspace(0,y_range,Ny)
    # Grids
    u = np.zeros(shape=(Nx, Ny), dtype=np.double)
    # Initial Guess
    if (guess == "zero"):
        pass
    elif (guess == "xy"):
        u = np.outer(dxs, dys)
    elif (guess == "random"):
        u = np.random.random((Nx, Ny))*2-1

    # Boundary Conditions
    bc = lambda x: np.sin(2*x) + np.sin(5*x) + np.sin(7*x)
    u[:,0] = 0
    u[:,Nx-1] = 0
    u[0,:] = bc(dxs)
    u[Ny-1,:] = 0
    return u

class solver:

    def __init__(self, u, tolerance):
        self.Ny = u.shape[0]
        self.Nx = u.shape[1]
        self.tolerance = tolerance
        self.u = u
        self.dx = np.pi/10
        self.dy = np.pi/10

    def res(self, u_next, j, i):
        value = (u_next[j + 1, i] - 2 * u_next[j, i] + u_next[j - 1, i]) / np.power(self.dy, 2) + \
                (u_next[j, i + 1] - 2 * u_next[j, i] + u_next[j, i - 1]) / np.power(self.dx, 2)

        return value

    def jacobi_iter(self, u_next, u):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j,i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def jacobiSOR_iter(self, u_next, u, w=0.8):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j,i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        u_next = (1-w)*u + w*u_next
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def gs_iter(self, u_next, u):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j,i] = self.res(u_next, j, i)
                u_next[j,i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u_next[j - 1, i] + u_next[j, i - 1])
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def gsSOR_iter(self, u_next, u, w=1.5):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j,i] = self.res(u_next, j, i)
                u_next[j,i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u_next[j - 1, i] + u_next[j, i - 1])
        u_next = (1-w)*u + w*u_next
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def solve(self, method):
        iter = None
        if (method == "jacobi"):
            iter = self.jacobi_iter
        elif (method == "gs"):
            iter = self.gs_iter
        elif (method == "jacobiSOR"):
            iter = self.jacobiSOR_iter
        elif (method == "gsSOR"):
            iter = self.gsSOR_iter

        res_array = []
        rmax = 10
        k = 0
        u_next = self.u.copy()
        u = self.u.copy()

        while rmax > self.tolerance:
            k += 1
            u_next, rmax = iter(u_next, u)
            res_array.append(rmax)
            # update u
            u = u_next.copy()
        return u, res_array

def compare_acc(result1, result2):
    return np.sum(np.sum(np.power(result1-result2,2)))

if __name__ == '__main__':
    tolerance = 1e-5
    solver_zero = solver(generate_u(guess="zero"), tolerance)
    solver_xy = solver(generate_u(guess="random"), tolerance)
    solver_random = solver(generate_u(guess="xy"), tolerance)
    # Question 1
    #jacobi_zero, res_zero_jb = solver_zero.solve("jacobi")
    #gs_zero, res_zero_gs = solver_zero.solve("gs")
    #jacobi_random, res_random_jb = solver_random.solve("jacobi")
    gs_random, res_random_gs = solver_zero.solve("gs")
    #jacobi_xy, res_xy_jb = solver_xy.solve("jacobi")
    #gs_xy, res_xy_gs = solver_xy.solve("gs")

    #print("Initial guess are zeros. Iterations: Jacobi {}, GS {}".format(len(res_zero_jb), 0))
    #print("Initial guess are randoms. Iterations: Jacobi {}, GS {}".format(len(res_random_jb), len(res_random_gs)))
    #print("Initial guess are xy. Iterations: Jacobi {}, GS {}".format(len(res_xy_jb), len(res_xy_gs)))
    # Question 2
    #jacobi_zero_sor, res_zero_jbsor = solver_zero.solve("jacobiSOR")
    gsSOR_random, resSOR_random_gs = solver_zero.solve("gsSOR")
    #print("Initial guess are zeros. Iterations: JacobiSOR {}, GS {}".format(len(res_zero_jb),0))
    print("Initial guess are randoms. Iterations: GS {}, GSSOR {}".format(len(res_random_gs), len(resSOR_random_gs)))



    

