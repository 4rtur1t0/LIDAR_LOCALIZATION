import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tools.simulate import mod_2pi


class GraphSLAM():
    def __init__(self, vertices, edges, ground_truth=None):
        self.ground_truth = ground_truth
        self.vertices = vertices
        self.edges = edges
        self.x = []
        self.H = []
        self.b = []
        self.indexesH = []
        self.init_x()

    def init_x(self):
        """
        :return:
        """
        N = len(self.vertices)
        self.x = np.zeros(3*N)
        for i in range(N):
            self.x[i*3] = self.vertices[i].x[0]
            self.x[i*3 + 1] = self.vertices[i].x[1]
            self.x[i*3 + 2] = self.vertices[i].x[2]
        return

    def optimize(self):
        """
        Perform iterations until dx is low
        :return:
        """
        self.init_x()
        while True:
            self.compute_hessian()
            # compute directly by inverting H (the information matrix)
            dx = self.solve_deltax()
            if np.linalg.norm(dx) < 0.01:
                break
        # unfix H
        self.H[0:3, 0:3] -= np.eye(3)
        # now resize H, removing row and column to compute
        n = len(self.x)
        self.plotH()
        # uncertainties relative to pose 0
        H2 = self.H[3:n, 3:n]
        self.S = np.linalg.inv(H2)
        self.plotS()
        self.plot_uncertainty()
        return self.x, self.H

    def compute_hessian(self):
        N = len(self.vertices)
        self.H = np.zeros((3*N, 3*N))
        self.b = np.zeros(3*N)
        # Function F to be minimized
        F = 0
        for edge in self.edges:
            Aij = edge.Aij(self.x)
            Bij = edge.Bij(self.x)
            eij = edge.eij(self.x)
            Oij = edge.Oij
            Hii = np.dot(Aij.T, np.dot(Oij, Aij))
            Hij = np.dot(Aij.T, np.dot(Oij, Bij))
            Hji = np.dot(Bij.T, np.dot(Oij, Aij))
            Hjj = np.dot(Bij.T, np.dot(Oij, Bij))
            bi = np.dot(Aij.T, np.dot(Oij, eij))
            bj = np.dot(Bij.T, np.dot(Oij, eij))
            # compute F based on the errors eij
            F += np.dot(eij.T, np.dot(Oij, eij))
            # update +=Hii, Hij, Hji, and Hjj
            # update +=bi, +=bj
            # update in full H
            self.update_Hb(edge.i, edge.j, Hii, Hij, Hji, Hjj, bi, bj)
            # self.plotH()
        self.H[0:3, 0:3] += np.eye(3)
        print('F is: ', F)
        # self.plotH()
        return self.H, self.b

    def plotH(self):
        plt.figure()
        plt.matshow(self.H)
        plt.show()

    def plotS(self):
        plt.figure()
        plt.matshow(self.S)
        plt.show()

    def update_Hb(self, i, j, Hii, Hij, Hji, Hjj, bi, bj):
        self.H[3*i:3*i + 3, 3*i:3*i + 3] += Hii
        self.H[3*i:3*i + 3, 3*j:3*j + 3] += Hij
        self.H[3*j:3*j + 3, 3*i:3*i + 3] += Hji
        self.H[3*j:3*j + 3, 3*j:3*j + 3] += Hjj
        self.b[3*i:3*i+3] += bi
        self.b[3*j:3*j+3] += bj
        return

    def solve_deltax(self):
        # dx = solve(H, -b)
        dx = -np.dot(np.linalg.inv(self.H), self.b)
        self.x = self.x + dx
        print('Norm update is: ', np.linalg.norm(dx))
        return dx

    def plot(self, title='UNTITLED'):
        plt.figure()
        if self.ground_truth is not None:
            N = len(self.ground_truth)
            gt = np.zeros((N, 3))
            for i in range(N):
                gt[i, 0] = self.ground_truth[i].x[0]
                gt[i, 1] = self.ground_truth[i].x[1]
                gt[i, 2] = self.ground_truth[i].x[2]
            plt.plot(gt[:, 0], gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
        N = len(self.vertices)
        x = np.zeros((N, 3))
        for i in range(N):
            x[i, 0] = self.x[3*i]
            x[i, 1] = self.x[3*i + 1]
            x[i, 2] = self.x[3*i + 2]

        plt.plot(x[:, 0], x[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
        for edge in self.edges:
            i = int(edge.i)
            j = int(edge.j)
            a = [x[i, 0], x[j, 0]]
            b = [x[i, 1], x[j, 1]]
            plt.plot(a, b, color='black', linestyle='dotted', marker='o', markerfacecolor='black', markersize=12)
        plt.title(title)
        plt.show(block=True)

    def plot_uncertainty(self, title='UNTITLED'):
        # plt.figure()
        fig, ax = plt.subplots()
        if self.ground_truth is not None:
            N = len(self.ground_truth)
            gt = np.zeros((N, 3))
            for i in range(N):
                gt[i, 0] = self.ground_truth[i].x[0]
                gt[i, 1] = self.ground_truth[i].x[1]
                gt[i, 2] = self.ground_truth[i].x[2]
            plt.plot(gt[:, 0], gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
        # plot current solution
        N = len(self.vertices)
        x = np.zeros((N, 3))
        for i in range(N):
            x[i, 0] = self.x[3*i]
            x[i, 1] = self.x[3*i + 1]
            x[i, 2] = self.x[3*i + 2]
        plt.plot(x[:, 0], x[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)

        # remove first row and column
        for i in range(N-1):
            sx2 = self.S[3 * i, 3 * i]
            sy2 = self.S[3 * i + 1, 3 * i + 1]
            mux = self.x[3 * (i+1)]
            muy = self.x[3 * (i+1) + 1]
            muth = self.x[3 * (i+1) + 2]
            ellipse = Ellipse(xy=(mux, muy), width=2*np.sqrt(sx2), height=2*np.sqrt(sy2), angle=np.rad2deg(muth))
            ellipse.set_facecolor('none')
            ellipse.set_alpha(0.5)
            ellipse.set_facecolor('blue')
            ax.add_artist(ellipse)

        plt.title(title)
        plt.show(block=True)

    # def plotH(self):
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.imshow(self.H, cmap='hot', interpolation='nearest')
    #     plt.show()

class Vertex():
    def __init__(self, i=0, x=[]):
        self.i = i
        self.x = x

    def fromstring(self, string):
        if string.startswith("VERTEX2"):
            st = string.split()
            self.i = int(st[1])
            self.x = np.array([float(st[2]), float(st[3]), float(st[4])])
            return True
        else:
            return False


class Edge():
    def __init__(self, i=0, j=0, zij=[], Oij=[]):
        self.i = i
        self.j = j
        self.zij = zij
        self.Oij = Oij

    def fromstring(self, string):
        if string.startswith("EDGE2"):
            st = string.split()
            self.i = int(st[1])
            self.j = int(st[2])
            self.zij = np.array([float(st[3]), float(st[4]), float(st[5])])
            self.Oij = np.zeros((3, 3))
            self.Oij[0, 0] = float(st[6])
            self.Oij[1, 1] = float(st[7])
            self.Oij[2, 2] = float(st[8])
            return True
        else:
            return False

    def Aij(self, x):
        i = self.i
        j = self.j
        ti = x[3*i:3*i + 3]
        tj = x[3*j:3*j + 3]
        tij = self.zij
        xi = ti[0]
        yi = ti[1]
        thi = ti[2]
        xj = tj[0]
        yj = tj[1]
        thij = tij[2]
        Aij = np.array([[-np.cos(thi + thij), -np.sin(thi + thij), np.sin(thi + thij) * (xi - xj) - np.cos(thi + thij) * (yi - yj)],
                        [np.sin(thi + thij), -np.cos(thi + thij), np.cos(thi + thij) * (xi - xj) + np.sin(thi + thij) * (yi - yj)],
                        [0, 0, -1]])
        return Aij

    def Bij(self, x):
        i = self.i
        ti = x[3*i:3*i+3]
        tij = self.zij
        thi = ti[2]
        thij = tij[2]
        Bij = np.array([[np.cos(thi + thij), np.sin(thi + thij), 0],
                        [-np.sin(thi + thij), np.cos(thi + thij), 0],
                        [0, 0, 1]])
        return Bij

    def eij(self, x):
        """
        Compute error, given, i, j
        xi
        xj
        zij =
        :return:
        """
        i = self.i
        j = self.j
        ti = x[3 * i:3 * i + 3]
        tj = x[3 * j:3 * j + 3]
        tij = self.zij
        xi = ti[0]
        yi = ti[1]
        thi = ti[2]
        xj = tj[0]
        yj = tj[1]
        thj = tj[2]
        xij = tij[0]
        yij = tij[1]
        thij = tij[2]
        eij = np.zeros(3)
        eij[0] = -np.cos(thij)*(xij + np.cos(thi)*(xi-xj) + np.sin(thi)*(yi-yj))-np.sin(thij)*(yij + np.cos(thi)*(yi - yj) - np.sin(thi)*(xi - xj))
        eij[1] = np.sin(thij)*(xij + np.cos(thi)*(xi-xj) + np.sin(thi)*(yi - yj)) - np.cos(thij)*(yij + np.cos(thi)*(yi - yj) - np.sin(thi)*(xi - xj))
        eij[2] = thj-thi-thij
        eij[2] = mod_2pi(eij[2])
        return eij


