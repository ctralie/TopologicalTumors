import numpy as np

def get_shape_hist(X, n_shells, r_max, center=True):
    """
    Compute a shape histogram, counting points distributed
    in concentric spherical shells centered at the origin [1]

    [1] Ankerst 1999 "3D Shape Histograms for Similarity Search and 
    Classification in Spatial Databases"

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    n_shells: float
        Number of concentric shells
    r_max: float
        Maximum radius
    center: bool
        Whether to move the point cloud to its center
    
    Returns
    -------
    ndarray(n_shells): Shape histogram
    """
    if center:
        # Move to centroid
        X = X - np.mean(X, axis=0, keepdims=True)
    hist = np.zeros(n_shells)
    dists = np.sqrt(np.sum(X**2, axis=1))
    rs = np.linspace(0, r_max, n_shells+1)
    for i in range(n_shells):
        hist[i] = np.sum((dists >= rs[i])*(dists < rs[i+1]))
    return hist
    
def get_shape_shell_hist(X, n_shells, r_max, subdiv=1, center=True):
    """
    Create shape histogram with concentric spherical shells and
    sectors within each shell, sorted in decreasing order of 
    number of points [1]

    [1] Ankerst 1999 "3D Shape Histograms for Similarity Search and 
    Classification in Spatial Databases"

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    n_shells: float
        Number of concentric shells
    subdiv: int
        Number of subdivisions to perform on a regular icosahedron to
        obtain sample points.  Number of sectors is 2 + (20*4^(subdiv))/2
    r_max: float
        Maximum radius
    center: bool
        Whether to move the point cloud to its center
    
    Returns
    -------
    ndarray(n_shells*n_sectors) Shape shell histogram
    """
    if center:
        X = X - np.mean(X, axis=0, keepdims=True)
    # First sample points on the sphere by doing a regular subdivision
    # of an icosahedron
    from pymeshlab import MeshSet
    ms = MeshSet()
    ms.create_sphere(radius=1, subdiv=subdiv)
    S = ms.mesh(0).vertex_matrix()
    n_sectors = S.shape[0] # Num sectors is num points sampled on the sphere
    #Create a 2D histogram that is n_shells x n_sectors
    hist = np.zeros((n_shells, n_sectors))
    dists = np.sqrt(np.sum(X**2, axis=1))
    rs = np.linspace(0, r_max, n_shells+1)
    for i in range(0, n_shells):
        XSub = X[(dists >= rs[i])*(dists < rs[i+1]), :]
        scores = XSub.dot(S.T)
        idx = np.argmax(scores, 1)
        for k in range(n_sectors):
            hist[i, k] = np.sum(idx == k)
    hist = np.sort(hist, axis=1)
    return hist.flatten() #Flatten the 2D histogram to a 1D array

def get_shape_pca_hist(X, n_shells, r_max, center=True):
    """
    Create shape histogram with concentric spherical shells and 
    compute the PCA eigenvalues in each shell

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    n_shells: float
        Number of concentric shells
    r_max: float
        Maximum radius
    center: bool
        Whether to move the point cloud to its center
    
    Returns
    -------
    ndarray(n_shells*3): PCA Histogram
    """
    from sklearn.decomposition import PCA
    if center:
        X = X - np.mean(X, axis=0, keepdims=True)
    pca = PCA(n_components=3)
    #Create a 2D histogram, with 3 eigenvalues for each shell
    hist = np.zeros((n_shells, 3))
    dists = np.sqrt(np.sum(X**2, axis=1))
    rs = np.linspace(0, r_max, n_shells+1)
    for i in range(n_shells):
        XSub = X[(dists >= rs[i])*(dists < rs[i+1]), :]
        if XSub.size > 0:
            pca.fit(XSub)
            hist[i, :] = np.sqrt(pca.singular_values_)
    return hist.flatten()

def get_d2_hist(X, d_max, n_bins, n_samples):
    """
    Create shape histogram of the pairwise Euclidean distances between
    randomly sampled points in the point cloud [2]

    [2] Osada 2002 "Shape Distributions"

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    d_max: float
        Maximum pairwise distance to consider
    n_bins: int
        Number of bins in the histogram
    n_samples: int
        Number of random samples to take
    
    Returns
    -------
    ndarray(n_bins): D2 histogram
    """
    N = X.shape[0]
    X1 = X[np.random.random_integers(0, N-1, n_samples), :]
    X2 = X[np.random.random_integers(0, N-1, n_samples), :]
    d = np.sqrt(np.sum((X1-X2)**2, axis=1))
    return np.histogram(d, bins=n_bins, range=(0, d_max))[0]

def get_a3_hist(X, n_bins, n_samples):
    """
    Create shape histogram of the angles between randomly sampled
    triples of points [2]

    [2] Osada 2002 "Shape Distributions"

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    n_bins: int
        Number of bins in the histogram
    n_samples: int
        Number of random samples to take
    
    Returns
    -------
    ndarray(n_bins): D2 histogram
    """
    N = X.shape[0]
    X1 = X[np.random.random_integers(0, N-1, n_samples), :]
    X2 = X[np.random.random_integers(0, N-1, n_samples), :]  
    X3 = X[np.random.random_integers(0, N-1, n_samples), :] 
    dV1 = X2 - X1
    dV2 = X3 - X1
    denom = np.sqrt(np.sum(dV1**2, axis=1))
    denom[denom <= 0] = 1
    dV1 = dV1/denom[:, None]
    denom = np.sqrt(np.sum(dV2**2, axis=1))
    denom[denom <= 0] = 1
    dV2 = dV2/denom[:, None]
    dots = np.sum(dV1*dV2, axis=1)
    dots[dots < -1] = -1
    dots[dots > 1] = 1
    angles = np.arccos(dots)
    return np.histogram(angles, bins=n_bins, range=(0, np.pi))[0]

def get_egi(X, T, subdiv=1, sigma=None):
    """
    Compute the extended Gaussian image [3] by aligning a mesh to its 
    principal axes and then binning the normals

    [3] Horn 1984: "Extended Gaussian Images"

    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    T: ndarray(M, 3, dtype=int)
        Triangle indices; used to compute normals
    subdiv: int
        Number of subdivisions to perform on a regular icosahedron to
        obtain sampled normals.  Number of sectors is 2 + (20*4^(subdiv))/2
    sigma: float
        Standard deviation in angle to use for normal weighting function when creating
        the histogram.  If unspecified, make it the average angle between adjacent
        points in the mesh
    
    Returns
    -------
    hist: ndarray(2 + (20*4^(subdiv))/2)
        Histogram of values
    S: ndarray(2 + (20*4^(subdiv))/2, 3)
        Sphere samples, with radii of points proportional to the histogram (for visualization)
    ST: ndarray(12*4^(subdiv), 3)
        Triangles of sphere mesh (for visualization)
    """
    from sklearn.decomposition import PCA
    from pymeshlab import MeshSet
    from utils3d import get_vertex_normals
    ## Step 1: Get normal directions
    ms = MeshSet()
    ms.create_sphere(radius=1, subdiv=subdiv)
    S = ms.mesh(0).vertex_matrix()
    ST = ms.mesh(0).face_matrix()
    n_sphere = S.shape[0]
    
    ## Step 1b: Compute average angle between neighbors on the spherical mesh
    if not sigma:
        from utils3d import get_edges
        I, J = get_edges(S, ST)
        dot = np.sum(S[I, :]*S[J, :], axis=1)
        dot[dot < -1] = -1
        dot[dot > 1] = 1
        sigma = np.mean(np.arccos(dot))

    ## Step 2: Align the normals to the principal axes of the points
    pca = PCA(n_components=3)
    pca.fit(X)
    N = pca.transform(get_vertex_normals(X, T))
    hist = np.zeros(n_sphere)
    scores = N.dot(S.T)
    scores[scores < -1] = -1
    scores[scores > 1] = 1
    scores = np.arccos(scores)
    scores = np.exp(-scores**2/(2*sigma**2))
    hist = np.sum(scores, axis=0)
    hist /= np.sum(hist)
    for k in range(n_sphere):
        S[k, :] *= hist[k]
    return hist, S, ST

def get_spin_image(X, n_angles, extent, dim):
    """
    Compute the spin image [4]

    [4] Johnson 1997 "Spin-images: a representation for 3-D surface matching"
    
    Parameters
    ----------
    X: ndarray(N, 3)
        Point samples
    n_angles: int
        Number of angles to spin
    extent: float
        Maximum radial extent of the swept plane
    dim: int
        Resolution of the spin image
    
    Returns
    -------
    ndarray(dim, dim)
        The spin image
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    X = X - np.mean(X, axis=0, keepdims=True)
    hist = np.zeros((dim, dim))

    ## Step 1: Align image to its principal axes
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    
    ## Step 2: Keep the first axis fixed, then rotate the other two axes
    for a in np.linspace(0, 2*np.pi, n_angles+1)[0:n_angles]:
        [c, s] = [np.cos(a), np.sin(a)]
        R = np.array([[c, s], [-s, c]])
        Y = np.array(X)
        Y[:, 1::] = Y[:, 1::].dot(R)
        hist += np.histogram2d(Y[:, 0], Y[:, 1], dim, [[-extent, extent], [-extent, extent]])[0]
    hist = hist/np.sum(hist) #Normalize before returning
    return hist