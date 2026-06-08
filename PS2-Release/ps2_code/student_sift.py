import numpy as np
import cv2
from ps2_code.student_harris import get_gradients

def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.
    
    Helpful functions: np.arctan2
    
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.

    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    magnitudes = np.sqrt(dx.astype(np.float64)**2 + dy.astype(np.float64)**2)
    orientations = np.arctan2(dy, dx).astype(np.float64)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations

def get_feat_vec(x,y,magnitudes, orientations,feature_width):
    """
    This function returns the feature vector for a specific interest point.

    To start with, you might want to use normalized patches as your
    local feature. This should be simple to code and works OK.

    However, to get full credit you will need to implement the more effective
    SIFT descriptor (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells where each cell is a square of feature_width/4 * feature_width/4 pixels
        It is simply the terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be appended
        to the feature vector left to right then row by row (reading order).
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.

    Useful function: np.histogram

    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    height, width = magnitudes.shape
    cell = feature_width // 4

    int_x = int(x)
    int_y = int(y)

    r0, r1 = int_y - (feature_width // 2), int_y + (feature_width // 2)
    c0, c1 = int_x - (feature_width // 2), int_x + (feature_width // 2)

    magnitude_window = np.zeros((feature_width, feature_width), dtype=np.float64)
    orietation_window = np.zeros((feature_width, feature_width), dtype=np.float64)

    rr0, rr1 = max(r0, 0), min(r1, height)
    cc0, cc1 = max(c0, 0), min(c1, width)
    wr0, wr1 = rr0 - r0, rr0 - r0 + (rr1 - rr0)
    wc0, wc1 = cc0 - c0, cc0 - c0 + (cc1 - cc0)

    if rr1 > rr0 and cc1 > cc0:
        magnitude_window[wr0:wr1, wc0:wc1] = magnitudes[rr0:rr1, cc0:cc1]
        orietation_window[wr0:wr1, wc0:wc1] = orientations[rr0:rr1, cc0:cc1]

    edges = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0.0, np.pi/4,  np.pi/2,  3*np.pi/4,  np.pi], dtype=np.float64)

    hists = []
    for i in range(4):
        rstart, rend = i * cell, (i + 1) * cell
        for j in range(4):
            cstart, cend = j * cell, (j + 1) * cell
            m = magnitude_window[rstart:rend, cstart:cend].ravel()
            o = orietation_window[rstart:rend, cstart:cend].ravel()
            hist, _ = np.histogram(o, bins=edges, weights=m)
            hists.append(hist.astype(np.float64))

    fv = np.concatenate(hists, axis=0)
    eps = 1e-12
    n = np.linalg.norm(fv)
    if n > 0: fv = fv / (n + eps)
    fv = np.power(fv, 0.9)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before.

    Useful function: get_gradients, get_magnitudes_and_orientations


    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    assert image.ndim == 2, 'Image must be grayscale'
    fvs = []
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    dx, dy = get_gradients(image)
    magnutides, orientations = get_magnitudes_and_orientations(dx, dy)
    
    for x_, y_ in zip(x,y):
        fv = get_feat_vec(float(x_), float(y_), magnutides, orientations, feature_width)
        fvs.append(fv)
        
    fvs = np.vstack(fvs) if len(fvs) > 0 else np.zeros((0, 128), dtype=np.float64)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs
