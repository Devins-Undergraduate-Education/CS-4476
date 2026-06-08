import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used later (in get_interest_points for calculating
    image gradients and a second moment matrix).
    You can call this function to get the 2D gaussian filter.

    Hints:
    1) Make sure the value sum to 1
    2) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """

    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    gausKern = cv2.getGaussianKernel(ksize, sigma) # ksize, 1
    kernel = gausKern @ gausKern.T
    kernel = (kernel / np.sum(kernel))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return kernel

def my_filter2D(image, filter, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Hints:
        Padding width should be half of the filter's shape (correspondingly)
        The conv_image shape should be same as the input image
        Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c),
                depending if image is grayscale or colored
    -   filter: filter that will be used in the convolution with shape (a,b)
    -   bias: An bias value added to every output

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################

    image = np.asarray(image)
    isColor = (image.ndim == 3)
    
    height, width = filter.shape
    pad_h, pad_w = height // 2, width // 2
    
    k = np.flipud(np.fliplr(filter))
    
    if isColor: 
        H, W, C = image.shape
        padding = cv2.copymakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        out = np.zeros_like(image, dtype=np.float64)
        for c in range(C):
            channel = padding[:, :, c]
            for y in range(H):
                for x in range(W):
                    out[y, x, c] = np.sum(channel[y:y+height, x:x+width] * k) + bias
        conv_image = out.astype(np.float64)
    else: 
        H, W = image.shape
        padding = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        out = np.zeros_like(image, dtype=np.float64)
        for y in range(H):
            for x in range(W):
                out[y, x] = np.sum(padding[y:y+height, x:x+width] * k) + bias
        conv_image = out.astype(np.float64)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter,
    which is of shape (3, 3). Sobel filters can be used to approximate the image
    gradient, and it will be a different filter for the x and y directions.

    Helpful functions: my_filter2D from above

    Args:
    -   image: A numpy array of shape (m,n) containing the image

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction

    Note: Remember that the image gradient in the x-direction corresponds to vertical edge detection and vice versa for y.
    """

    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    if image.ndim == 3:
        img = image.mean(axis=2)
    else:
        img = image
    img = img.astype(np.float64)
    
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    
    ix = my_filter2D(img, sobel_x)
    iy = my_filter2D(img, sobel_y)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    height, width = image.shape[:2]
    div_2 = window_size // 2
    
    mask = (x >= div_2) & (x <= width - div_2 - 1) & (y >= div_2) & (x <= height - div_2 - 1)
    return x[mask], y[mask], c[mask]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.
    Second moments, AKA the variance, provide a measure of how spread out the values are in a distribution.
    These moments are computed by convolving the image gradients with a Gaussian filter.

    Helpful functions: my_filter2D, get_gaussian_kernel

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    gaus = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix * ix, gaus)
    sy2 = my_filter2D(iy * iy, gaus)
    sxsy = my_filter2D(ix * iy, gaus)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments function below, calculate corner resposne.

    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    det = sx2 * sy2 - (sxsy ** 2)
    R = det - alpha * ((sx2 + sy2) ** 2)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression.
    Take a matrix and return a matrix of the same size but only the max values in a neighborhood that are not zero.
    We also do not want very small local maxima so remove all values that are below the median for the original R matrix.

    The input to this function is corner response matrix and the output is a filtered version of this
    matrix, where some of the responses have been set to 0.

    Helpful functions: scipy.ndimage.filters.maximum_filter

    Args:
    -   R: numpy nd-array of shape (m, n)
    -   neighborhood_size: int, the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero
    """

    R_local_pts = None

    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    
    lmax = maximum_filter(R, size=neighborhood_size, mode='constant')
    R_local_pts = np.where((R == lmax) & (R > np.median(R)), R, 0)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts


def get_interest_points(image, n_pts = 1500):
    """
    Using your helper functions above, implement the Harris corner detector
    (See Szeliski 4.1.1). You will calculate the image gradients and second moments,
    use these to determine pixels with high corner response, and filter them via
    non maximum suppression and removing border values. You should return the
    top n_pts based on confidence score.

    Helpful functions:
        get_gradients, second_moments, corner_response, non_max_suppression, remove_border_vals

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer, number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """

    x, y, R_local_pts, confidences = None, None, None, None

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################

    if image.ndim == 3: gray = image.mean(axis=2).astype(np.float64)
    else: gray = image.astype(np.float64)
    
    ix, iy = get_gradients(gray)
    sx2, sy2, sxsy = second_moments(ix, iy, ksize=7, sigma=10)
    
    alpha = 0.05
    R = corner_response(sx2, sy2, sxsy, alpha)
    
    R_local_pts = non_max_suppression(R, neighborhood_size=7)
    y, x = np.nonzero(R_local_pts)
    confidences = R_local_pts[y, x]
    
    if len(confidences) == 0:
        return np.array([]), np.array([]), R_local_pts, np.array([])
    
    order = np.argsort(confidences)[::-1]
    x = x[order]
    y = y[order]
    confidences = confidences[order]
    
    x, y, confidences = remove_border_vals(image, x, y, confidences, window_size=16)
    n = min(n_pts, x.shape[0])
    x = x[:n].astype(np.int32)
    y = y[:n].astype(np.int32)
    confidences = confidences[:n].astype(np.float64)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return x,y, R_local_pts, confidences
