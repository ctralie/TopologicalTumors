import numpy as np

def gauss3d(w):
    """
    Get a 3D Gaussian Kernel

    Parameters
    ----------
    w: int
        Width of kernel
    
    Returns
    -------
    ndarray(w, w, w)
        Kernel
    """
    sigma = 2*(w/2)/6 # https://github.com/scikit-image/scikit-image/blob/f13b28d75a7400eb94eb0cff6f5a7ac03aa4eb8a/skimage/transform/pyramids.py#L90
    kernel = np.arange(-(w//2), w//2+1)
    kernel = np.exp(-kernel**2/(2*sigma**2))
    kernel = kernel[None, None, :]*kernel[None, :, None]*kernel[:, None, None]
    kernel /= np.sum(kernel)
    return kernel
