import numpy as np
import matplotlib.pyplot as plt

def prob_1_1(N):
    """
    Args: N: the number of trials. 
    Returns: arr: array of rolls.
    """

    ### START CODE HERE ###
    # casting to int via int(...) does not work
    # add '1' due to floor function
    arr = (np.floor(np.random.rand(N) * 6) + 1).astype(int)
    ### END CODE HERE ###

    return arr

def prob_1_2(y):
    """
    Args: y: numpy array. 
    Returns: z: numpy array of shape (new_size,2).
    """

    ### START CODE HERE ###
    z = y.reshape(-1,2)
    ### END CODE HERE ###

    return z


def prob_1_3(z):
    """
    Args: z: numpy array of shape (3,2).
    Returns: x: max value in z.
    r: row index of x.
    c: column index of x.
    """

    ### START CODE HERE ###
    x = int(np.max(z)) # cast to int to avoid 'np.int64(...)'
    row, col = np.where(z == x) # np.where(...) returns 'r, c'
    r = int(row[0])
    c = int(col[0])
    ### END CODE HERE ###

    return (x, r, c)


def prob_1_4(v):
    """
    Args: v: numpy array. 
    Returns: x: number of 1's in v.
    """

    ### START CODE HERE ###
    x = np.sum(v == 1)
    ### END CODE HERE ###

    return x


print(prob_1_1(10))

y = np.array([11, 22, 33, 44, 55, 66])
print(prob_1_2(y))

z = y.reshape(-1, 2)
print(prob_1_3(z))

v = np.array([1, 4, 7, 1, 2, 6, 8, 1, 9])
print(prob_1_4(v))