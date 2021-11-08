import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class unsteadyConduction:
    def __init__(self,Nx, Nt, L, T, m):
        self.Nx = Nx
        self.Nt = Nt
        self.L = L
        self.m = m
        self.x = np.linspace(0, L, Nx+1) # mesh points in space
        self.t = np.linspace(0, T, Nt+1) # mesh points in time
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        self.F = self.dt/self.dx**2
        self.u = np.zeros(Nx+1) # unknown u at a new time level
        self.u_n = np.zeros(Nx+1) # u at the previous time level

    def steadyChecker(self):
        if (self.F > .5):
            print(self.F)
            raise Exception("The scheme is not stable")

    def ResetSolver(self):
        """
        Reset the solver
        Make sure apply this step after solving an equation
        :return:
        """
        self.u = np.zeros(self.Nx+1)
        self.u_n = np.zeros(self.Nx+1)
    def I(self, x):
        """
        The initial condition function I(x) = Sin(mx)
        :param x:
        :return:
        """
        return np.sin(self.m*x)

    def setUpIC(self):
        """
        Set up the initial condition u(x,0) = I(x)
        :return:
        """
        for i in range(0, self.Nx+1):
            self.u_n[i] = self.I(self.x[i])

    def FESolver(self):
        """
        The forward euler method to solve the equation
        :return:
        """
        self.setUpIC()
        self.steadyChecker()
        for n in range(0, self.Nt):
            for i in range(1, self.Nx):
                self.u[i] = self.u_n[i] + self.F*(self.u_n[i+1] - 2*self.u_n[i] + self.u_n[i-1])

            # Boundary Conditions
            self.u[0] = 0
            self.u[self.Nx] = 0

            # update before each step
            self.u_n, self.u = self.u, self.u_n

if __name__ == '__main__':
    Nx = 100
    Nt = 10000
    L = 2*np.pi
    T = 10
    FE2 = unsteadyConduction(Nx, Nt, L, T, 2)
    FE2.FESolver()
    x = FE2.x
    t = FE2.t
    u_n = FE2.u_n
    print(x.shape, t.shape, u_n.shape)
    plt.plot(x,u_n)
    plt.show()
    plt.plot()





