import numpy as np
import matplotlib.pyplot as plt

def get_digits(foldername):
    """
    Load all of the digits 0-9 in from some folder

    Parameters
    ----------
    foldername: string
        Path to folder containing 0.png, 1.png, ..., 9.png
    
    Returns 
    -------
        X: ndarray(n_examples, 28*28)
            Array of digits
        y: ndarray(n_examples, dtype=int)
            Number of each digit
    """
    res = 28
    X = []
    Y = []
    for num in range(10):
        I = plt.imread("{}/{}.png".format(foldername, num))
        row = 0
        col = 0
        while row < I.shape[0]:
            col = 0
            while col < I.shape[1]:
                img = I[row:row+res, col:col+res]
                if np.sum(img) > 0:
                    X.append(img.flatten())
                    Y.append(num)
                col += res
            row += res
    return 1-np.array(X), np.array(Y, dtype=int)

def randomly_transform(I, dt, dr):
    import skimage
    rot = np.random.randint(-dr, dr+1)
    trans = np.random.randint(-dt, dt+1, 2)
    J = skimage.transform.rotate(I, rot)
    tform = skimage.transform.AffineTransform(translation=trans)
    J = skimage.transform.warp(J, inverse_map=tform.inverse)
    return J