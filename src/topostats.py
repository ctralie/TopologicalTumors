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
    B = crop_binary_volume(B) # Crop to region of interest for speed
    V = fftconvolve(B, kernel, mode='full')
    c = CubicalComplex(top_dimensional_cells=V)
    pers = c.persistence()
    return gudhi2sktda(pers)

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
    return [I[np.isfinite(I[:, 1]), :] for I in PDs]