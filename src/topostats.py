import numpy as np

def gudhi2sktda(pers):
    Is = [[], [], []]
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    return [np.array(I) for I in Is]

def get_alpha_filtration_3d(X):
    """
    A wrapper around GUDHI for computing persistence diagrams of
    3D alpha filtrations on a point cloud

    Parameters
    ----------
    X: ndarray(N, 3)
        Points on the point cloud
    
    Returns
    -------
    list of 3 ndarray(N, 2)
        List of persistence diagrams in H0, H1, and H2 
    """
    from gudhi import AlphaComplex
    alpha_complex = AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree()
    pers = simplex_tree.persistence()
    Is = gudhi2sktda(pers)
    return [np.sqrt(I) for I in Is]

def get_binary_kernel_cubical_filtration(B, kernel):
    """
    Compute the cubical filtration of a kernel applied to
    a 3D binary image

    Parameters
    ----------
    B: ndarray(M, N, L)
        Binary volumetric function
    kernel: ndarray(m, n, l)
        Kernel to apply

    Returns
    -------
    list of 3 ndarray(N, 2)
        List of persistence diagrams in H0, H1, and H2 
    """
    from scipy.signal import fftconvolve
    from utils3d import crop_binary_volume
    from gudhi.cubical_complex import CubicalComplex
    ret = [np.array([]), np.array([]), np.array([])]
    if B.size > 0:
        B = crop_binary_volume(B) # Crop to region of interest for speed
        if B.size > 0:
            V = fftconvolve(B, kernel, mode='full')
            c = CubicalComplex(top_dimensional_cells=V)
            pers = c.persistence()
            ret = gudhi2sktda(pers)
    return ret

def remove_infinite(PDs):
    """
    Remove infinite points from persistence diagrams

    Parameters
    ----------
    list of ndarray(Ni, 2)
        Persistence diagrams
    
    Returns
    list of ndarray(<=Ni, 2)
        Persistence diagrams with infinite pairs removed
    """
    ret = []
    for I in PDs:
        if I.size > 0:
            ret.append(I[np.isfinite(I[:, 1]), :])
        else:
            ret.append(I)
    return ret

def get_persim_stack(PDs, pimgr):
    """
    Compute a stack of persistence images on a set of
    persistence diagrams

    Parameters
    ----------
    PDs: list of M ndarray(Ni, 2)
        Persistence diagrams
    pimgr: scipy.PersistenceImager
        Persistance imager object that creates dxd images
        from persistence diagrams
    
    Returns
    ndarray(M*3, d, d)
        Stack of filtered persistence images
    """
    stack = []
    PDs = remove_infinite(PDs)
    for PD in PDs:
        I = np.array(pimgr.transform(PD))
        stack.append(I)
    return np.array(stack)

def get_kernel_persim_stack(B, kernels, pimgr):
    """
    Compute the sublevelset filtrations of binary 3D images
    after they have been convolved with a set of kernels, and
    compute persistence images for each diagram for each kernel

    Parameters
    ----------
    B: ndarray(N, L, P)
        Binary volumetric function
    kernels: list of M ndarray(k, k)
        Kernels to apply before sublevelset filtrations
    pimgr: scipy.PersistenceImager
        Persistance imager object that creates dxd images
        from persistence diagrams
    
    Returns
    ndarray(M*3, d, d)
        Stack of filtered persistence images

    """
    PDs = []
    for kernel in kernels:
        PDs += get_binary_kernel_cubical_filtration(B, kernel)
    return get_persim_stack(PDs, pimgr)