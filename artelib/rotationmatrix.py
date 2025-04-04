#!/usr/bin/env python
# encoding: utf-8
"""
The Rotation Matrix class
@Authors: Arturo Gil
@Time: July 2022
"""
import numpy as np
from artelib import euler, quaternion, homogeneousmatrix, vector
import matplotlib.pyplot as plt


class RotationMatrix():
    def __init__(self, *args):
        if len(args) == 0:
            self.array = np.eye(3)
        elif len(args) > 0:
            orientation = args[0]
            # self.array = None
            # constructor from a np array
            if isinstance(orientation, np.ndarray):
                self.array = orientation
            elif isinstance(orientation, list):
                self.array = np.array(orientation)
            elif isinstance(orientation, int):
                self.array = np.eye(orientation)
            elif isinstance(orientation, euler.Euler):
                self.array = orientation.R()
            elif isinstance(orientation, quaternion.Quaternion):
                self.array = orientation.R()
            # copy constructor
            elif isinstance(orientation, RotationMatrix):
                self.array = orientation.toarray()
            else:
                raise Exception

    def __str__(self):
        return str(self.array)

    def __getitem__(self, item):
        return self.array[item[0], item[1]]

    def print_nice(self, precision=3):
        temp_array = self.array
        th = 0.01
        idx = np.abs(temp_array) < th
        temp_array[idx] = 0
        print(np.array_str(self.array, precision=precision, suppress_small=True))

    def print(self):
        self.print_nice()

    def toarray(self):
        return self.array

    def inv(self):
        return RotationMatrix(self.array.T)

    def T(self):
        """
        transpose, as in numpy
        """
        return RotationMatrix(self.array.T)

    def det(self):
        return np.linalg.det(self.array)

    def R(self):
        return self

    def Q(self):
        q = self.rot2quaternion()
        return quaternion.Quaternion(qw=q.qw, qx=q.qx, qy=q.qy, qz=q.qz)

    def euler(self):
        eul = self.rot2euler()
        return euler.Euler(eul[0]), euler.Euler(eul[1])

    def homogeneous(self):
        return homogeneousmatrix.HomogeneousMatrix(np.zeros(3), self)

    def __mul__(self, other):
        """
        Multiply matrices or multiply a matrix and a vector.
        """
        if isinstance(other, RotationMatrix):
            R = np.dot(self.array, other.array)
            return RotationMatrix(R)
        elif isinstance(other, vector.Vector):
            u = np.dot(self.array, other.array)
            return vector.Vector(u)

    def rot2euler(self):
        """
        Conversion from the rotation matrix R to Euler angles.
        The XYZ convention in mobile axes is assumed.
        """
        R = self.array
        th = np.abs(np.abs(R[0, 2]) - 1.0)
        R[0, 2] = min(R[0, 2], 1)
        R[0, 2] = max(R[0, 2], -1)

        # caso no degenerado
        if th > 0.0001:
            beta1 = np.arcsin(R[0, 2])
            beta2 = np.pi - beta1
            s1 = np.sign(np.cos(beta1))
            s2 = np.sign(np.cos(beta2))
            alpha1 = np.arctan2(-s1 * R[1][2], s1 * R[2][2])
            gamma1 = np.arctan2(-s1 * R[0][1], s1 * R[0][0])
            alpha2 = np.arctan2(-s2 * R[1][2], s2 * R[2][2])
            gamma2 = np.arctan2(-s2 * R[0][1], s2 * R[0][0])
        # degenerate case
        else:
            print('CAUTION: rot2euler detected a degenerate solution when computing the Euler angles.')
            alpha1 = 0
            alpha2 = np.pi
            beta1 = np.arcsin(R[0, 2])
            if beta1 > 0:
                beta2 = np.pi / 2
                gamma1 = np.arctan2(R[1][0], R[1][1])
                gamma2 = np.arctan2(R[1][0], R[1][1]) - alpha2
            else:
                beta2 = -np.pi / 2
                gamma1 = np.arctan2(-R[1][0], R[1][1])
                gamma2 = np.arctan2(-R[1][0], R[1][1]) - alpha2
        # finally normalize to +-pi
        e1 = normalize_angles([alpha1, beta1, gamma1])
        e2 = normalize_angles([alpha2, beta2, gamma2])
        return euler.Euler(e1), euler.Euler(e2)

    def rot2quaternion(self):
        """
        rot2quaternion(R)
        Computes a quaternion Q from a rotation matrix R.

        This implementation has been translated from The Robotics Toolbox for Matlab (Peter  Corke),
        https://github.com/petercorke/spatial-math

        CAUTION: R is a matrix with some noise due to floating point errors. For example, the determinant of R may no be
        exactly = 1.0 always. As a result, given R and R_ (a close noisy matrix), the resulting quaternions Q and Q_ may
        have a difference in their signs. This poses no problem, since the slerp formula considers the case in which
        the distance cos(Q1*Q_) is negative and changes it sign (please, see slerp).

        There are a number of techniques to retrieve the closest rotation matrix R given a noisy matrix R1.
        In the method below, this consideration is not taken into account, however, the trace tr is considered always
        positive. The resulting quaternion, as stated before, may have a difference in sign.

        On Closed-Form Formulas for the 3D Nearest Rotation Matrix Problem. Soheil Sarabandi, Arya Shabani, Josep M. Porta and
        Federico Thomas.

        http://www.iri.upc.edu/files/scidoc/2288-On-Closed-Form-Formulas-for-the-3D-Nearest-Rotation-Matrix-Problem.pdf

        """
        R = self.array[0:3, 0:3]
        tr = np.trace(R) + 1
        # caution: tr should not be negative
        tr = max(0.0, tr)
        s = np.sqrt(tr) / 2.0
        kx = R[2, 1] - R[1, 2]  # Oz - Ay
        ky = R[0, 2] - R[2, 0]  # Ax - Nz
        kz = R[1, 0] - R[0, 1]  # Ny - Ox

        # equation(7)
        k = np.argmax(np.diag(R))
        if k == 0:  # Nx dominates
            kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1  # Nx - Oy - Az + 1
            ky1 = R[1, 0] + R[0, 1]  # Ny + Ox
            kz1 = R[2, 0] + R[0, 2]  # Nz + Ax
            sgn = mod_sign(kx)
        elif k == 1:  # Oy dominates
            kx1 = R[1, 0] + R[0, 1]  # % Ny + Ox
            ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1  # Oy - Nx - Az + 1
            kz1 = R[2, 1] + R[1, 2]  # % Oz + Ay
            sgn = mod_sign(ky)
        elif k == 2:  # Az dominates
            kx1 = R[2, 0] + R[0, 2]  # Nz + Ax
            ky1 = R[2, 1] + R[1, 2]  # Oz + Ay
            kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1  # Az - Nx - Oy + 1
            # add = (kz >= 0)
            sgn = mod_sign(kz)
        # equation(8)
        kx = kx + sgn * kx1
        ky = ky + sgn * ky1
        kz = kz + sgn * kz1

        nm = np.linalg.norm([kx, ky, kz])
        if nm == 0:
            # handle the special case of null quaternion
            # Q = np.array([1, 0, 0, 0])
            Q = quaternion.Quaternion(qw=1, qx=0, qy=0, qz=0)
        else:
            v = np.dot(np.sqrt(1 - s ** 2) / nm, np.array([kx, ky, kz]))  # equation(10)
            # Q = np.hstack((s, v))
            Q = quaternion.Quaternion(qw=s, qx=v[0], qy=v[1], qz=v[2])
        return Q


    def plot(self, title='Rotation Matrix', block=True):
        """
        Plot the rotation matrix as 2D or 3D vectors
        """
        n = self.array.shape[0]
        fig = plt.figure()
        if n == 2:
            # origin = np.array([[0, 0], [0, 0]])  # origin point
            plt.quiver((0, 0), (0, 0), self.array[0, :], self.array[1, :], color=['red', 'green'], angles='xy',
                       scale_units='xy', scale=1)
            plt.draw()
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.xlabel('X')
            plt.ylabel('Y')
        elif n == 3:
            ax = fig.add_subplot(projection='3d')
            # first drawing the "-" . Next drawing two lines for each head ">"
            colors = ['red', 'green', 'blue', 'red', 'red', 'green', 'green', 'blue', 'blue']
            ax.view_init(15, 35)
            # plot identity
            I = np.eye(3)
            ax.quiver(0, 0, 0, I[0, :], I[1, :], I[2, :], color=colors, linestyle='dashed', linewidth=3)
            ax.quiver(0, 0, 0, self.array[0, :], self.array[1, :], self.array[2, :], color=colors, linewidth=3)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        plt.title(title)
        plt.show(block=block)


def R2(theta):
    """
    A 2x2 rotation matrix.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return RotationMatrix(R)


def Rx(theta):
    """
    A fundamental rotation along the X axis.
    """
    R = np.array([[1,       0,                    0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return RotationMatrix(R)


def Ry(theta):
    """
    A fundamental rotation along the Y axis.
    """
    R = np.array([[np.cos(theta), 0,    np.sin(theta)],
                  [0,             1,                0],
                  [-np.sin(theta),  0,     np.cos(theta)]])
    return RotationMatrix(R)


def Rz(theta):
    """
    A fundamental rotation along the Z axis.
    """
    R = np.array([[np.cos(theta), -np.sin(theta),  0],
                  [np.sin(theta),  np.cos(theta),  0],
                  [0,              0,        1]])
    return RotationMatrix(R)


def mod_sign(x):
    """
       modified  version of sign() function as per   the    paper
        sign(x) = 1 if x >= 0
    """
    if x >= 0:
        return 1
    else:
        return -1


def normalize_angles(eul):
    """
    Normalize angles in array to [-pi, pi]
    """
    e = []
    for i in range(len(eul)):
        e.append(np.arctan2(np.sin(eul[i]), np.cos(eul[i])))
    return e
