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

def laplacian3d(w, normalize=True):
    """
    Get a 3D Laplacian Kernel

    Parameters
    ----------
    w: int
        Width of kernel
    normalize: bool
        Whether to make sure the convolution with a binary image
        would give a sum in the range [-1, 1]
    
    Returns
    -------
    ndarray(w, w, w)
        Kernel
    """
    sigma = 2*(w/2)/6 # https://github.com/scikit-image/scikit-image/blob/f13b28d75a7400eb94eb0cff6f5a7ac03aa4eb8a/skimage/transform/pyramids.py#L90
    pix = np.arange(-(w//2), w//2+1)
    x, y, z = np.meshgrid(pix, pix, pix, indexing='ij')
    r_sqr = x**2 + y**2 + z**2
    kernel = (r_sqr/sigma**2 - 1)*np.exp(-r_sqr/(2*sigma**2))
    if normalize:
        neg = -1*np.sum(kernel[kernel < 0])
        pos = np.sum(kernel[kernel > 0])
        kernel = kernel/max(neg, pos)
    return kernel

def get_random_3d_kernels(k, N):
    """
    Compute N kxkxk kernels whose entries are sampled from
    a random Gaussian, and normalize each one so that the range
    of the output of their convolution with a binary image is
    within [-1, 1]

    Parameters
    ----------
    k: int
        Dimension of kernels
    N: int
        Number of kernels

    Returns
    -------
    ndarray(N, k, k, k)
        The N normalized random kxkxk kernels
    """
    kernels = []
    for i in range(N):
        kernel = np.random.randn(k, k, k)
        neg = -1*np.sum(kernel[kernel < 0])
        pos = np.sum(kernel[kernel > 0])
        kernel = kernel/max(neg, pos)
        kernels.append(kernel)
    return kernels
