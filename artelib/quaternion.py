#!/usr/bin/env python
# encoding: utf-8
"""
The Quaternion orientation class

@Authors: Arturo Gil
@Time: April 2021
"""
import numpy as np
from artelib import euler, rotationmatrix, homogeneousmatrix
# from artelib import quaternion2rot, q2euler


class Quaternion():
    def __init__(self, qx, qy, qz, qw):
       # self.array = np.array(array)
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def R(self):
        return rotationmatrix.RotationMatrix(self.quaternion2rot())

    def homogeneous(self):
        return homogeneousmatrix.HomogeneousMatrix(np.zeros(3), self.R())

    def euler(self):
        """
        Convert Quaternion to Euler angles XYZ
        """
        return self.q2euler()

    def Q(self):
        return self

    def __str__(self):
        return str([self.qx, self.qy, self.qz, self.qw])

    def toarray(self):
        return np.array([self.qx, self.qy, self.qz, self.qw])

    def __add__(self, Q):
        return Quaternion(qx=self.qx + Q.qx,
                          qy=self.qy + Q.qy,
                          qz=self.qz + Q.qz,
                          qw=self.qw + Q.qw)

    def __sub__(self, Q):
        return Quaternion(qx=self.qx - Q.qx,
                          qy=self.qy - Q.qy,
                          qz=self.qz - Q.qz,
                          qw=self.qw - Q.qw)

    def dot(self, Q):
        """
        quaternion dot product
        """
        s = self.qx * Q.qx + self.qy * Q.qy + self.qz * Q.qz + self.qw * Q.qw
        return s

    def qconj(self):
        """
        quaternion conjugate
        """
        return Quaternion(qx=-self.qx,
                          qy=-self.qy,
                          qz=-self.qz,
                          qw=self.qw)

    def __mul__(self, Q):
        """
        quaternion product
        """
        if isinstance(Q, Quaternion):
            s1 = self.qw
            s2 = Q.qw
            v1 = np.array([self.qx, self.qy, self.qz])
            v2 = np.array([Q.qx, Q.qy, Q.qz])
            qw = s1*s2-np.dot(v1, v2)
            qv = s1*v2 + s2*v1 + np.cross(v1, v2)
            return Quaternion(qx=qv[0], qy=qv[1], qz=qv[2], qw=qw)
        # this assumes that the rightmost element is a float
        elif isinstance(Q, int) or isinstance(Q, float):
            s = Q # scalar
            q = np.dot(s, [self.qx, self.qy, self.qz, self.qw])
            return Quaternion(qx=q[0], qy=q[1], qz=q[2], qw=q[3])
        else:
            raise Exception('Quaternion product does not support the leftmost operand')

    def __truediv__(self, Q):
        """
        Division of quaternion by constant
        """
        if isinstance(Q, int) or isinstance(Q, float):
            # scalar, consider the input Q as scalar
            s = Q
            q = np.dot(1/s, [self.qx, self.qy, self.qz, self.qw])
            return Quaternion(qx=q[0], qy=q[1], qz=[2], qw=[3])
        else:
            raise Exception('Quaternion division by scalar does not support the leftmost operand')

    def quaternion2rot(self):
        qw = self.qw
        qx = self.qx
        qy = self.qy
        qz = self.qz
        R = np.eye(3)
        R[0, 0] = 1 - 2 * qy ** 2 - 2 * qz ** 2
        R[0, 1] = 2 * qx * qy - 2 * qz * qw
        R[0, 2] = 2 * qx * qz + 2 * qy * qw
        R[1, 0] = 2 * qx * qy + 2 * qz * qw
        R[1, 1] = 1 - 2 * qx ** 2 - 2 * qz ** 2
        R[1, 2] = 2 * qy * qz - 2 * qx * qw
        R[2, 0] = 2 * qx * qz - 2 * qy * qw
        R[2, 1] = 2 * qy * qz + 2 * qx * qw
        R[2, 2] = 1 - 2 * qx ** 2 - 2 * qy ** 2
        return R

    def q2euler(self):
        R = self.quaternion2rot()
        e1, e2 = R.euler()
        return e1, e2