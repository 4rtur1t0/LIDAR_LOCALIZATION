"""
A class
"""
import numpy as np
from scipy.spatial import KDTree

from config import PARAMETERS


class LoopClosing():
    def __init__(self, graphslam):
        """
        This class provides functions for loop-closing in a ICP context using LiDAR points.
        Though called DataAssociation, it really provides ways to find out whether the computation of the
        relative transformations (registration) between LiDAR pointclouds is correct.
        Actually, data associations are performed in a lazy manner: if we find a pose near the pose i in the current
        estimation, we will try to compute a transformation between the two pointclouds. A correct registration using
        a vanilla ICP algorithm depends upon an initial transformation between the two pointclouds. If this initial
        transformation is very noisy, the final estimation may be incorrect. In order to avoid incorrect transformation,
         we have implemented the following process. The method is called loop_closing_triangle proposes that, for a
         given observation at time i:
        a) finds other robot pose j within r1 and r2 radii in terms of travelled distance. In this way, j is close to i
        and possesses a low uncertainty. The tranformation Tij is computed
        b) then looks for a different pose k within radius r3 (a loop closing radidus) that, as well is far in terms of
        travel distance. The radius r3 absorbs large error in loop closing.
        c) As a result, we have triplets of poses (i, j, k) considering that the indexes i and j must be close
        (i.e. close in time) and k is far. The transformations Tjk and Tik are computed.
        c) For each triplet, it must be: Tij*Tjk*Tik.inv()=Tii=I, the identity. Due to errors, it may be different
        from the identity. The resulting matrix I is then converted to I.pos() and I.euler() and checked to find p = 0,
        and abg=0 approximately. If the transformation differs largely from I, the observations are discarded.
        If Tii = I approximately, we add to the graph Tij Tjk and Tik.

        Care must be taken, since the pose estimations in graphslam estimate the position/orientation of the GPS (which
        differs from the location of the LiDAR).
        """
        self.graphslam = graphslam
        self.positions = None

    def find_feasible_triplets(self):
        """
        Find loop closure candidates across the entire trajectory.

        For each pose i:
            - Find a pose j with distance R:  min_travel_distance <  R(i, j) < max_travel_distance.
            - Find a pose k so that: min_travel_distance < R(i,k) < max_travel_distance
                                     and min_travel_distance < R(j,k) < max_travel_distance

        Args:
            poses (np.ndarray): Nx3 or Nx2 array of poses (x, y, theta optional).
            parameters in parameters.yaml (radii to find candidates).

        Returns:
            list: A list of triplets (i, j, k).
        """
        positions = self.graphslam.get_solution_positions()
        if len(positions) == 0:
            return None
        # precompute travel distances to easily find the travel distance between nodes i and j in the graph (in xy position)
        travel_distances = self.compute_travel_distances(positions)
        # sample i at travel distances
        d_sample_i = PARAMETERS.config.get('loop_closing').get('d_sample_i') # 5.0 # meters
        print('SAMPLING i poses with a relative distance of d_sample_i: ', d_sample_i)
        sampled_i = [0]
        for i in range(1, len(positions)):
            d_rel = travel_distances[i]-travel_distances[sampled_i[-1]]
            if d_rel > d_sample_i:
                sampled_i.append(i)
        print('Sampled for loop closing at a total of i poses: ', len(sampled_i))
        # Build KDTree for fast spatial queries (for loop closing)
        tree = KDTree(positions[:, :2])  # Use (x, y) positions
        triplets_global = []
        # for each i find j. The index j is found at a distance
        # for i in range(0, len(positions), step_index):
        for i in sampled_i:
            # find an index j close to i (within r_min and r_max) and close in the sequence index
            triplet = self.find_j_k_within_radii(tree=tree, positions=positions, travel_distances=travel_distances,
                                                 i=i)
            if triplet is not None:
                triplets_global.append(triplet)
        # flatten
        triplets_global = [item for sublist in triplets_global for item in sublist]
        return triplets_global

    def compute_travel_distances(self, positions):
        """
        Compute cumulative travel distance for each pose
        """
        distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        return np.insert(np.cumsum(distances), 0, 0)

    def find_j_k_within_radii(self, tree, positions, travel_distances, i):
        """
        Finds a candidate j at a distance  r_close1 < d < r_close2
        It should be at a close index, so that the error in poses from i to j is low (id est, low uncertainty).
        The first j is selected for this purpose.
        Also find a candidate k that both meets:
        - r_lc (radius for loop closing): may be large, depending on the expected uncertainty in loop closing.
        - r_traveled: may be large, in order to select large loop closings (well separated in time). However, small
        loop closings may be also beneficial.
        If different k meet these requirements, generate aThe number of triplets
        """
        # find candidates for j within r_close. The index j must be
        r_close1 = PARAMETERS.config.get('loop_closing').get('r_close1') #0.7
        r_close2 = PARAMETERS.config.get('loop_closing').get('r_close2') #1.5 # this is the actual loop closing distance
        # find candidates for long loopclosing. Find candidates within r_lc that have travelled more than r_lc_travel_distance
        # if r_lc > r_lc --> there exists the possibility to include small loop closings
        # if r_lc_travel_distance > r_lc, then only large loop closings will be included
        r_lc = PARAMETERS.config.get('loop_closing').get('r_lc') # 3.5
        r_lc_travel_distance = PARAMETERS.config.get('loop_closing').get('r_lc_travel_distance') #3.0
        # num_triplets = 5
        num_triplets = PARAMETERS.config.get('loop_closing').get('num_triplets')
        # for clarity, we ask the tree for candidates
        # neighbors_in_r2 = tree.query_ball_point(positions[i, :2], r_close2)
        j_n = None
        # select only a first close candidate j within a distance of r1 and travel distance within r1 and r2
        # this candidate should be close in the sequence of observations (odometry, sm)
        # start at i and find index j that meet the criteria
        for j in range(i+1, len(positions)):
            d = travel_distances[j] - travel_distances[i]
            if r_close1 < d < r_close2: #(d > r_close1) and (d < r_close2):
                j_n = j
                break
        # select another candidate with a travel distance larger than r3
        neighbors_in_r_lc = tree.query_ball_point(positions[i, :2], r_lc)
        k_n = []
        # obtain k for a long travel distance (try to perform long loop closings)
        for k in neighbors_in_r_lc:
            d = travel_distances[k] - travel_distances[i]
            if k > i and (d > r_lc_travel_distance):
                k_n.append(k)
        num_triplets = min(num_triplets, len(k_n))
        # this is a uniform choice. IDEA: could try to include longer loop closings with more probability.
        # however, closer loop closings are also beneficial in this particular case
        k_n = np.random.choice(k_n, num_triplets, replace=False)
        result_triplets = []
        for k in k_n:
            if (j_n is not None) and (k is not None):
                result_triplets.append([i, j_n, k])
        return result_triplets

    def check_triplet_transforms(self, triplet_transforms):
        """
        A better loop closing procedure. Given the current pose and index i (current_index):
                a) Find a number of past robot poses inside a radius_threshold.
                b) Chose a candidate j randomly. Find another candidate k. The distance in the indexes in j and k < d_index
                c) Compute observations using ICP for Tij and Tik.
                d) Compute Tij*Tjk*(Tik)^(-1)=I, find the error in position and orientation in I to filter the validity
                   of Tij and Tik. Tjk should be low in uncertainty, since it depends on consecutive observations.
                e) Add the observations with add_loop_closing_restrictions.
        Still, of course, sometimes, the measurement found using ICP may be wrong, in this case, it is less probable that
         both Tij and Tik have errors that can cancel each other. As a result, this is a nice manner to filter out observations.
        """
        result_triplet_transforms = []
        for triplet_transform in triplet_transforms:
            print('Checking loop closing triplet: ', triplet_transform)
            # the transformation computed from the LiDAR using ICP. Caution: expressed in the LiDAR ref frame.
            Tij = triplet_transform[3]
            Tjk = triplet_transform[4]
            Tik = triplet_transform[5]
            # compute circle transformation
            I = Tij * Tjk * Tik.inv()
            print('Found loop closing triplet I: ', I)
            if self.check_distances(I):
                print(10*'#')
                print('FOUND CONSISTENT OBSERVATIONS!')
                print('Adding loop closing observations to list!.')
                print(10 * '#')
                # result_triplet_transforms.append([i, j, k, Tij, Tjk, Tik])
                result_triplet_transforms.append(triplet_transform)
        return result_triplet_transforms

    def check_distances(self, I):
        dp = np.linalg.norm(I.pos())
        da1 = np.linalg.norm(I.euler()[0].abg)
        da2 = np.linalg.norm(I.euler()[1].abg)
        da = min([da1, da2])
        print('Found triangle loop closing distances: ', dp, da)
        if dp < 0.1 and da < 0.05:
            print('I is OK')
            return True
        print('FOUND INCONSISTENT LOOP CLOSING TRIPLET: DISCARDING!!!!!!!')
        return False

