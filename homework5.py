import numpy as np
import matplotlib.pyplot as plt


def generate_u(guess, dx=np.pi / 10, dy=np.pi / 10, x_range=2 * np.pi, y_range=2 * np.pi):
    Nx = int(x_range / dx) + 1
    Ny = int(y_range / dy) + 1
    dxs = np.linspace(0, x_range, Nx)
    dys = np.linspace(0, y_range, Ny)
    # Grids
    u = np.zeros(shape=(Nx, Ny), dtype=np.double)
    # Initial Guess
    if (guess == "zero"):
        pass
    elif (guess == "xy"):
        u = np.outer(dxs, dys)
    elif (guess == "random"):
        u = np.random.random((Nx, Ny)) * 2 - 1

    # Boundary Conditions
    bc = lambda x: np.sin(2 * x) + np.sin(5 * x) + np.sin(7 * x)
    u[:, 0] = 0
    u[:, Nx - 1] = 0
    u[0, :] = bc(dxs)
    u[Ny - 1, :] = 0
    return u


class solver:

    def __init__(self, u, tolerance):
        self.Ny = u.shape[0]
        self.Nx = u.shape[1]
        self.tolerance = tolerance
        self.u = u
        self.dx = np.pi / 10
        self.dy = np.pi / 10

    def res(self, u_next, j, i):
        value = (u_next[j + 1, i] - 2 * u_next[j, i] + u_next[j - 1, i]) / np.power(self.dy, 2) + \
                (u_next[j, i + 1] - 2 * u_next[j, i] + u_next[j, i - 1]) / np.power(self.dx, 2)

        return value

    def jacobi_iter(self, u_next, u, w=None, k=None):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def jacobiSOR_iter(self, u_next, u, w=1, k=None):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        u_next = (1 - w) * u + w * u_next
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def gs_iter(self, u_next, u, w=None, k=None):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u_next[j - 1, i] + u_next[j, i - 1])
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def gsSOR_iter(self, u_next, u, w=1, k=None):
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u_next[j - 1, i] + u_next[j, i - 1])
        u_next = (1 - w) * u + w * u_next
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def jacobi_over_under(self, u_next, u, w, k):
        w = w[0] if k % 2 == 1 else w[1]
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        u_next = (1 - w) * u + w * u_next
        r_max = np.max(np.abs(r))

        return u_next, r_max

    def jacobi_tp(self, u_next, u, w, k):
        w = w[0] if k % 3 ==  1 else w[1] if k % 3 == 2 else w[2]
        r = np.zeros_like(u)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                r[j, i] = self.res(u_next, j, i)
                u_next[j, i] = 0.25 * (u[j + 1, i] + u[j, i + 1] + u[j - 1, i] + u[j, i - 1])
        u_next = (1 - w) * u + w * u_next
        r_max = np.max(np.abs(r))
        return u_next, r_max

    def solve(self, method, w):
        iter = None
        if (method == "jacobi"):
            iter = self.jacobi_iter
        elif (method == "gs"):
            iter = self.gs_iter
        elif (method == "jacobiSOR"):
            iter = self.jacobiSOR_iter
        elif (method == "gsSOR"):
            iter = self.gsSOR_iter
        elif (method == "jacobiou"):
            iter = self.jacobi_over_under
        elif (method == "jacobitp"):
            iter = self.jacobi_tp

        res_array = []
        rmax = 10
        k = 0
        u_next = self.u.copy()
        u = self.u.copy()
        while rmax > self.tolerance:
            k += 1
            u_next, rmax = iter(u_next, u, w=w, k=k)
            res_array.append(rmax)
            # update u
            u = u_next.copy()

        return u, res_array


def compare_acc(result1, result2):
    return np.sum(np.sum(np.power(result1 - result2, 2)))


def Q1Results():
    # Write results into the file
    f = open("hw5result/result_nums.txt", mode="a")
    methods = ["Jacobi", "GS"]
    for method in methods:
        # set up a figure third as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(1 / 3))
        fig.suptitle("Solution by {} Method with Different Initial Guess".format(method))
        tolerance = 1e-6
        guess = ["zero", "xy", "random"]
        for i in range(1, 4):
            igtype = solver(generate_u(guess=guess[i - 1]), tolerance)
            sol, res_array = igtype.solve(method.lower())
            f.write("{}-{}-{}\n".format(method, guess[i - 1], len(res_array)))
            # plot the solutions
            ax_sol = fig.add_subplot(2, 3, i, projection='3d')
            x = np.linspace(0, 2 * np.pi, 21)
            y = np.linspace(0, 2 * np.pi, 21)
            x, y = np.meshgrid(x, y)
            surf = ax_sol.plot_surface(x, y, sol)
            ax_sol.set_title("Initial Guess: {} ".format(guess[i - 1]))
            # plot the convergence
            ax_covergence = fig.add_subplot(2, 3, i + 3)
            iter_num = np.arange(1, len(res_array) + 1)
            ax_covergence.plot(iter_num, res_array)
            ax_covergence.set_xlabel("Iteration Numbers")
            ax_covergence.set_ylabel("Residual")

        plt.savefig("./hw5result/q1_{}.png".format(method))
    f.close()


def find_optimal(method):
    ws = []
    if method == "jacobiSOR":
        ws = [0.7, 0.8, 0.9, 1.0, 1.03]
    elif method == "gsSOR":
        ws = [0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8]

    guess = ["zero", "xy", "random"]
    tolerance = 1e-6
    f = open("hw5result/result_nums.txt", mode="w")
    f.write(method + "\n")

    fig = plt.figure(figsize=plt.figaspect(1 / 3))
    fig.suptitle("{} Method with Different Initial Guess".format(method))

    for i in range(1, 4):
        print("{} begins...".format(guess[i-1]))
        iter_nums = []
        igtype = solver(generate_u(guess=guess[i - 1]), tolerance)
        f.write(guess[i - 1] + "\n")
        for w in ws:
            try:
                sol, res_array = igtype.solve(method, w=w)
                iter_nums.append(len(res_array))
                f.write("w:{}-N:{}-".format(w, len(res_array)))
                print("w={}: N={}".format(w,len(res_array)), end=" ;")
            except RuntimeWarning:
                print("The method is not convergent.")
        ax = fig.add_subplot(1, 3, i)
        ax.set_title("Initial Guess: {} ".format(guess[i - 1]))
        ax.plot(ws, iter_nums)
        ax.set_xlabel("omega")
        ax.set_ylabel("iteration times")
        print("\n{} completed".format(guess[i-1]))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./hw5result/q2_{}.png".format(method))
    f.close()

def Jacobi_OU():
    wpairs = [[3,0.3], [0.5,1.8], [1.8, 0.5], [0.7,1.5], [1.5,0.7], [0.8, 1.2], [1.2, 0.8], ]
    guess = ["zero", "xy", "random"]
    tolerance = 1e-6
    f = open("hw5result/Jacobi_pairs.txt", mode="w")
    for i in range(1,4):
        print("{} begins...".format(guess[i - 1]))
        iter_nums = []
        igtype = solver(generate_u(guess=guess[i - 1]), tolerance)
        f.write(guess[i - 1] + "\n")
        for wpair in wpairs:
            sol, res_array = igtype.solve("jacobiou", w=wpair)
            iter_nums.append(len(res_array))
            f.write("w:{}-N:{}-".format(wpair, len(res_array)))
            print("w={}: N={}".format(wpair, len(res_array)), end=" ;")
    f.close()

def Jacobi_TP():
    wpairs = [[4,0.5,0.5],[3,0.8,0.5],[2,0.7,0.7]]
    guess = ["zero", "xy", "random"]
    tolerance = 1e-6
    f = open("hw5result/Jacobi_tp.txt", mode="w")
    for i in range(1,4):
        print("{} begins...".format(guess[i - 1]))
        iter_nums = []
        igtype = solver(generate_u(guess=guess[i - 1]), tolerance)
        f.write(guess[i - 1] + "\n")
        for wpair in wpairs:
            sol, res_array = igtype.solve("jacobitp", w=wpair)
            iter_nums.append(len(res_array))
            f.write("w:{}-N:{}-".format(wpair, len(res_array)))
            print("w={}: N={}".format(wpair, len(res_array)), end=" ;")
    f.close()

if __name__ == '__main__':
    Jacobi_TP()
