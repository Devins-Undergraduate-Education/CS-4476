import math
from typing import Tuple

import numpy as np

from ps3_code import fundamental_matrix, two_view_data
from ps3_code.least_squares_fundamental_matrix import solve_F


def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int representing the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here
    p_all_inliers = ind_prob_correct ** sample_size
    num_samples = math.ceil(math.log(1 - prob_success) / math.log(1 - p_all_inliers))
    ##############################

    return num_samples


def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix of shape (3, 3)
    -   x_1s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: float representing the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """
    inliers = None

    ##############################
    # TODO: Student code goes here
    epilines_right = (F.T @ x_0s.T).T  # (N, 3)

    numerators = np.abs(np.einsum('ij,ij->i', epilines_right, x_1s))
    denominators = np.linalg.norm(epilines_right[:, :2], axis=1)
    errors = numerators / denominators

    inliers = np.nonzero(errors < threshold)[0]
    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.
        4. You can use the `preprocess_data` function in `two_view_data` to make
           x_0s, and x_1s homogeneous.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None

    ##############################
    # TODO: Student code goes here
    sample_pts = 8
    threshold = 1.0
    confidence = 0.99
    inlier_ratio_guess = 0.5

    x_0s_arr = np.asarray(x_0s)
    x_1s_arr = np.asarray(x_1s)
    x_0 = np.hstack([x_0s_arr, np.ones((x_0s_arr.shape[0], 1), dtype=x_0s_arr.dtype)])
    x_1 = np.hstack([x_1s_arr, np.ones((x_1s_arr.shape[0], 1), dtype=x_1s_arr.dtype)])

    iterations = int(math.ceil(calculate_num_ransac_iterations(confidence, sample_pts, inlier_ratio_guess))) or 1

    best_inliers = np.array([], dtype=int)
    rng = np.random.default_rng()

    for _ in range(iterations):
        idx = rng.choice(x_0s_arr.shape[0], sample_pts, replace=False)
        F_cand = solve_F(x_0s_arr[idx], x_1s_arr[idx])
        inliers = find_inliers(x_0, F_cand, x_1, threshold)

        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_F = F_cand

            if best_inliers.size == x_0s_arr.shape[0]:
                break

    inliers_x_0 = x_0s_arr[best_inliers]
    inliers_x_1 = x_1s_arr[best_inliers]
    ##############################

    return best_F, inliers_x_0, inliers_x_1
