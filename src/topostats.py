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


##########################################################
##                  FRACTAL DIMENSION                   ##
##########################################################

def get_h0_total(X, alpha):
    """
    Compute the total persistence in H0 by quickly computing the 
    MST as a subset of the Delaunay complex
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    alpha: float
        Power to weight each edge length
    
    Returns
    -------
    float: Power weighted total persistence sum
    """
    from scipy import sparse
    from scipy.spatial import Delaunay
    from itertools import combinations

    N = X.shape[0]
    simplices = Delaunay(X).simplices
    I = np.array([])
    J = np.array([])
    V = np.array([])
    counts = np.array([])

    ## Step 1: Compute distances of all edges in the Delaunay complex
    for (i, j) in combinations(range(simplices.shape[1]), 2):
        Ik = simplices[:, i]
        Jk = simplices[:, j]
        Vk = np.sqrt(np.sum((X[Ik, :] - X[Jk, :])**2, axis=1))
        I = np.concatenate((I, Ik))
        J = np.concatenate((J, Jk))
        V = np.concatenate((V, Vk))
        counts = np.concatenate((counts, np.ones(Ik.size)))
    Ds = sparse.coo_matrix((V, (I, J)), shape=(N, N))
    counts = sparse.coo_matrix((counts, (I, J)), shape=(N, N))
    Ds = Ds/counts
    
    ## Step 2: Compute the minimum spanning tree to get H0
    tree = sparse.csgraph.minimum_spanning_tree(Ds)
    ds = np.array(tree[tree.nonzero()]).flatten()
    return np.sum(ds**alpha)
    
def get_h0_fractaldim(X, alpha=0.5, samples=100):
    """
    Compute the H0-based fractal dimension, according to the procedure in [1]
    
    [1] Fractal Dimension Estimation with Persistent Homology: A Comparative Study
    Jonathan Jaquette and Benjamin Schweinhart
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    alpha: float
        Power to weight each edge length
    
    Returns
    -------
    float: Fractal dimension estimate
    """
    from scipy.stats import linregress
    N = X.shape[0]
    Ns = 10**np.linspace(np.log10(N/10), np.log10(N), 100)
    Ns = np.array(np.floor(Ns), dtype=int)
    Es = []
    for n in Ns:
        Xn = X[np.random.permutation(N)[0:n], :]
        En = get_h0_total(Xn, alpha)
        Es.append(En)
    beta = linregress(np.log(Ns), np.log(Es)).slope
    return alpha/(1-beta)

def find_dist_thresh(X, n_target, tree=None, eps_initial=1e-3, n_iters=10, verbose=False):
    """
    Find the distance threshold at which n edges exist, using
    Golden sections search
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    n_target: int
        Target number of points
    tree: scipy.spatial.KDTree
        KDTree, if precomputed
    eps_initial: float
        Epsilon at which to start searching for interval endpoints
    n_inters: int
        Number of iterations of golden sections
    verbose: bool
        If true, print golden section search iterations
    
    Returns
    -------
    float: Epsilon value that gets close to n_target
    """
    from scipy.spatial import KDTree
    N = X.shape[0]
    if not tree:
        tree = KDTree(X)
    get_n = lambda eps: np.sum(tree.query_ball_point(X, eps, return_length=True))-N
    
    ## Step 1: Figure out interval endpoints
    a = eps_initial
    n_est = get_n(a)
    while n_est > n_target:
        a /= 2
        n_est = get_n(a)
    b = a
    while n_est < n_target:
        b *= 2
        n_est = get_n(b)
    
    ## Step 2: Run golden sections search
    gr = (np.sqrt(5) + 1)/2
    f = lambda eps: np.abs(get_n(eps)-n_target)
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    for i in range(n_iters):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        eps = (b + a)/2
        if verbose:
            print(eps, get_n(eps))
        if get_n(eps) == n_target:
            break
    return eps
    
def get_correlation_dimension(X, n_samples=100):
    """
    Compute the correlation dimension for fractal estimation
    
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud
    n_samples: int
        Number of distance thresholds to take in the estimation
    """
    from scipy.stats import linregress
    from scipy.spatial import KDTree
    N = X.shape[0]
    tree = KDTree(X)
    n1 = N**0.75
    n2 = 50*N

    eps1 = find_dist_thresh(X, n1, tree=tree)
    eps2 = find_dist_thresh(X, n2, tree=tree)
    print("eps1", eps1)
    print("eps2", eps2)
    all_eps = 2**np.linspace(np.log2(eps1), np.log2(eps2), n_samples)
    
    
    get_n = lambda eps: np.sum(tree.query_ball_point(X, eps, return_length=True))-N
    C = [get_n(eps) for eps in all_eps]    

    return linregress(np.log(all_eps), np.log(C)).slope

def get_correlation_dimension_grid(B, dmin, dmax, verbose=False):
    """
    Compute the fractal correlation dimension for a point cloud that's
    binned to a regular grid in 3D
    
    Parameters
    ----------
    B: ndarray(M, N, L)
        A volume of 1s and 0s.  1s are locations of points
    dmin: float
        Minimum distance threshold to use in correlation dimension regression
    dmax: float
        Maximum distance threshold to use in correlation dimension regression
    verbose: bool
        Whether to print progress information
    
    Returns
    -------
    float: Fractal dimension estimate
    """
    from scipy.stats import linregress
    
    ## Step 1: Devise grid offsets
    pix = np.arange(0, int(np.ceil(dmax+1)))
    dI, dJ, dK = np.meshgrid(pix, pix, pix, indexing='ij')
    dI, dJ, dK = dI.flatten(), dJ.flatten(), dK.flatten()
    dists = np.sqrt(dI**2+dJ**2+dK**2)
    valid = (dists <= dmax)*(dists > 0)
    dI = dI[valid]
    dJ = dJ[valid]
    dK = dK[valid]
    dists = dists[valid]
    dist_counts = {d:0 for d in np.unique(dists)}
    if verbose:
        print(dI.size, "neighbors")

    ## Step 2: Examine all grid offsets for valid neighbors
    Bi, Bj, Bk = np.meshgrid(*[np.arange(B.shape[k]) for k in range(3)], indexing='ij')
    Bi, Bj, Bk = Bi.flatten(), Bj.flatten(), Bk.flatten()
    all_dists = []
    for count, (di, dj, dk, dist) in enumerate(zip(dI, dJ, dK, dists)):
        if verbose and count%100 == 0:
            print(".", end="")
        idxI = Bi + di
        idxJ = Bj + dj
        idxK = Bk + dk
        valid =  (idxI >= 0)*(idxI < B.shape[0])
        valid *= (idxJ >= 0)*(idxJ < B.shape[1])
        valid *= (idxK >= 0)*(idxK < B.shape[2])
        idxI = idxI[valid]
        idxJ = idxJ[valid]
        idxK = idxK[valid]
        dist_counts[dist] += np.sum(B[idxI, idxJ, idxK])
    
    x = np.array(list(dist_counts.keys()))
    y = np.array([dist_counts[d] for d in x])
    y = np.cumsum(y)
    ## TODO: Does having more distances logarithmically spaced higher up bias the result?
    return linregress(np.log(x[x >= dmin]), np.log(y[x >= dmin])).slope