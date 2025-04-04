import numpy as np
from back_del.homogeneousmatrix import HomogeneousMatrix
from tools.euler import Euler


def mod_2pi(th):
    thetap = np.arctan2(np.sin(th), np.cos(th))
    return thetap


def simulate_odometry(odo0, edges, N):
    """
    Simulate odometry starting at odo0.
    Edges are the relative transformation between i and j.
    :param odo0:
    :param edges:
    :param N:
    :return:
    """
    odo = [np.array(odo0)]
    for k in range(N):
        i = int(edges[k][0])
        j = int(edges[k][1])
        if abs(j-i) > 1:
            break
        ti = np.array(odo[i])
        tij = np.array(edges[k][2:5])
        Ti = HomogeneousMatrix(np.hstack((ti[0:2], 0)), Euler([0, 0, ti[2]]))
        Tij = HomogeneousMatrix(np.hstack((tij[0:2], 0)), Euler([0, 0, tij[2]]))

        # compute Tj given Ti and the observation
        Tj = Ti*Tij
        tj = Tj.t2v()

        xb = tj[0]
        yb = tj[1]
        thetab = tj[2]
        thetab = mod_2pi(thetab)
        odo.append(np.array([xb, yb, thetab]))
    odo = np.array(odo)
    return odo


def simulate_icp(x_gt, observations):
    N = len(observations)
    edges = []
    sx = 0.05
    sy = 0.05
    sth = 0.02
    observation_matrix = np.array([1/sx**2, 1/sy**2, 1/sth**2, 0, 0, 0])
    for k in range(N):
        i = observations[k][0]
        j = observations[k][1]
        ti = np.hstack((x_gt[i, 0:2], 0))
        tj = np.hstack((x_gt[j, 0:2], 0))
        Ti = HomogeneousMatrix(ti, Euler([0, 0, x_gt[i, 2]]))
        Tj = HomogeneousMatrix(tj, Euler([0, 0, x_gt[j, 2]]))
        Tiinv = Ti.inv()
        zij = Tiinv*Tj
        zij = zij.t2v()
        zij = zij + np.array([np.random.normal(0, sx, 1)[0], np.random.normal(0, sy, 1)[0], np.random.normal(0, sth, 1)[0] ])
        # np.random.normal([0, 0, 0], [sx, sy, sth], 1)
        zij[2] = mod_2pi(zij[2])
        idx = np.array([int(i), int(j)])
        edges.append(np.hstack((idx, zij, observation_matrix)))
    edges = np.array(edges)
    return edges


def vertices_from_icp(odo0, edges):
    """
    Simulate odometry starting at odo0.
    Edges are the relative transformation between i and j.
    :param odo0:
    :param edges:
    :param N:
    :return:
    """
    odo = [np.array(odo0)]
    for k in range(len(edges)):
        i = int(edges[k][0])
        j = int(edges[k][1])
        if abs(j-i) > 1:
            break

        ti = np.array(odo[i])
        tij = edges[k][2]
        Ti = HomogeneousMatrix(np.hstack((ti[0:2], 0)), Euler([0, 0, ti[2]]))
        Tij = HomogeneousMatrix(np.hstack((tij[0:2], 0)), Euler([0, 0, tij[2]]))

        # compute Tj given Ti and the observation
        Tj = Ti*Tij
        tj = Tj.t2v()

        xb = tj[0]
        yb = tj[1]
        thetab = tj[2]
        thetab = mod_2pi(thetab)
        odo.append(np.array([xb, yb, thetab]))
    odo = np.array(odo)
    return odo



def relative_movements(odometry):
    """
    Transform from odometry to relative movements
    :param odo0:
    :param edges:
    :param N:
    :return:
    """
    edges = []
    for k in range(len(odometry)-1):
        odoi = odometry[k]
        odoj = odometry[k+1]
        Ti = HomogeneousMatrix(np.hstack((odoi[0:2], 0)), Euler([0, 0, odoi[2]]))
        Tj = HomogeneousMatrix(np.hstack((odoj[0:2], 0)), Euler([0, 0, odoj[2]]))

        # compute Tj given Ti and the observation
        # compute relative movement between frame Ti ant Tj
        Tij = Ti.inv()*Tj
        tij = Tij.t2v()

        xb = tij[0]
        yb = tij[1]
        thetab = tij[2]
        thetab = mod_2pi(thetab)
        edges.append(np.array([xb, yb, thetab]))
    edges = np.array(edges)
    return edges




