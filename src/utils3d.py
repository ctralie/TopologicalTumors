import numpy as np
import os
from scipy import sparse

def read_mesh(path):
    """
    Read in a mesh file

    Parameters
    ----------
    path: string
        Path to mesh file
    
    Returns
    -------
    vertices: ndarray(N, 3)
        Array of vertex positions
    faces: ndarray(M, 3)
        Indices into the vertex set of the triangle faces
    """
    from pymeshlab import MeshSet
    if not os.path.isfile(path):
        return None
    ms = MeshSet()
    ms.load_new_mesh(path)
    vertices = ms.mesh(0).vertex_matrix()
    faces = ms.mesh(0).face_matrix()
    return vertices, faces

def sample_by_area(VPos, ITris, npoints, colPoints = False):
    """
    Randomly sample points by area on a triangle mesh.  This function is
    extremely fast by using broadcasting/numpy operations in lieu of loops
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    npoints : int
        Number of points to sample
    colPoints : boolean (default True)
        Whether the points are along the columns or the rows
    
    Returns
    -------
    (Ps : NDArray (npoints, 3) array of sampled points, 
     Ns : Ndarray (npoints, 3) of normals at those points   )
    """
    ###Step 1: Compute cross product of all face triangles and use to compute
    #areas and normals (very similar to code used to compute vertex normals)

    #Vectors spanning two triangle edges
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    FNormals = np.cross(V1, V2)
    FAreas = np.sqrt(np.sum(FNormals**2, 1)).flatten()

    #Get rid of zero area faces and update points
    ITris = ITris[FAreas > 0, :]
    FNormals = FNormals[FAreas > 0, :]
    FAreas = FAreas[FAreas > 0]
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]

    #Compute normals
    NTris = ITris.shape[0]
    FNormals = FNormals/FAreas[:, None]
    FAreas = 0.5*FAreas
    FNormals = FNormals
    VNormals = np.zeros_like(VPos)
    VAreas = np.zeros(VPos.shape[0])
    for k in range(3):
        VNormals[ITris[:, k], :] += FAreas[:, None]*FNormals
        VAreas[ITris[:, k]] += FAreas
    #Normalize normals
    VAreas[VAreas == 0] = 1
    VNormals = VNormals / VAreas[:, None]

    ###Step 2: Randomly sample points based on areas
    FAreas = FAreas/np.sum(FAreas)
    AreasC = np.cumsum(FAreas)
    samples = np.sort(np.random.rand(npoints))
    #Figure out how many samples there are for each face
    FSamples = np.zeros(NTris, dtype=np.int32)
    fidx = 0
    for s in samples:
        while s > AreasC[fidx]:
            fidx += 1
        FSamples[fidx] += 1
    #Now initialize an array that stores the triangle sample indices
    tidx = np.zeros(npoints, dtype=np.int64)
    idx = 0
    for i in range(len(FSamples)):
        tidx[idx:idx+FSamples[i]] = i
        idx += FSamples[i]
    idx = 0

    #Vector used to determine if points need to be flipped across parallelogram
    V3 = P2 - P1
    V3 = V3/np.sqrt(np.sum(V3**2, 1))[:, None] #Normalize

    #Randomly sample points on each face
    #Generate random points uniformly in parallelogram
    u = np.random.rand(npoints, 1)
    v = np.random.rand(npoints, 1)
    Ps = u*V1[tidx, :] + P0[tidx, :]
    Ps += v*V2[tidx, :]
    #Flip over points which are on the other side of the triangle
    dP = Ps - P1[tidx, :]
    proj = np.sum(dP*V3[tidx, :], 1)
    dPPar = V3[tidx, :]*proj[:, None] #Parallel project onto edge
    dPPerp = dP - dPPar
    Qs = Ps - dPPerp
    dP0QSqr = np.sum((Qs - P0[tidx, :])**2, 1)
    dP0PSqr = np.sum((Ps - P0[tidx, :])**2, 1)
    idxreg = np.arange(npoints, dtype=np.int64)
    idxflip = idxreg[dP0QSqr < dP0PSqr]
    u[idxflip, :] = 1 - u[idxflip, :]
    v[idxflip, :] = 1 - v[idxflip, :]
    Ps[idxflip, :] = P0[tidx[idxflip], :] + u[idxflip, :]*V1[tidx[idxflip], :] + v[idxflip, :]*V2[tidx[idxflip], :]

    #Step 3: Compute normals of sampled points by barycentric interpolation
    Ns = u*VNormals[ITris[tidx, 1], :]
    Ns += v*VNormals[ITris[tidx, 2], :]
    Ns += (1-u-v)*VNormals[ITris[tidx, 0], :]

    if colPoints:
        return (Ps.T, Ns.T)
    return (Ps, Ns)

def get_indicator_submanifold(VPos, ITris, ind):
    """
    Return a triangle mesh that's a subset of the given mesh where
    an indicator function is 1
    
    Parameters
    ----------
    VPos: ndarray(N, 3)
        Vertex positions
    ITris: ndarray(T, 3, dtype=int)
        Triangle indices
    ind: ndarray(N)
        Indicator function on the vertices
    
    Parameters
    ----------
    VSub: ndarray(<=N, 3)
        Vertices of the submanifold,
    TSub: ndarray(<=T, 3)
        Triangles of the submanifold, indexed according to the vertex order
    """
    N = VPos.shape[0]
    idx = np.where(ind == 1)[0]
    VSub = VPos[idx, :]
    reindex = -1*np.ones(N, dtype=int)
    reindex[idx] = np.arange(idx.size)
    ITris = np.array(ITris, dtype=int)
    tris_used = reindex[ITris]
    tris_used = np.sum(tris_used > -1, axis=1) == 3
    TSub = ITris[tris_used, :]
    TSub = reindex[TSub]
    return VSub, TSub

def get_edges(VPos, ITris):
    """
    Given a list of triangles, return an array representing the edges
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    Returns: I, J
        Two parallel 1D arrays with indices of edges
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)
    for shift in range(3): 
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j] 
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i] 
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    return L.nonzero()

def get_vertex_neighbors(ITris):
    """
    Return a dictionary of vertex neighbors from a specification of
    the triangles in a mesh

    Parameters
    ----------
    ITris: ndarray(T, 3, dtype=int)
        Triangle indices
    
    Returns
    -------
    dictionary: int => set of int
        Neighbors of all vertices
    """
    neighbs = {}
    for tri in ITris:
        for idx in range(3):
            i, j, k = tri[idx], tri[(idx+1)%3], tri[(idx+2)%3]
            if not i in neighbs:
                neighbs[i] = set([])
            neighbs[i].add(j)
            neighbs[i].add(k)
    return neighbs

def get_vertex_normals(VPos, ITris):
    """
    Compute the vertex normals as weighted sums of adjacent
    face normals

    Parameters
    ----------
    VPos: ndarray(N, 3)
        Vertex positions
    ITris: ndarray(T, 3, dtype=int)
        Triangle indices
    
    Returns
    -------
    ndarray(N, 3)
        Vertex normals
    """
    #Vectors spanning two triangle edges
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    FNormals = np.cross(V1, V2)
    FAreas = np.sqrt(np.sum(FNormals**2, 1)).flatten()

    #Get rid of zero area faces and update points
    ITris = ITris[FAreas > 0, :]
    FNormals = FNormals[FAreas > 0, :]
    FAreas = FAreas[FAreas > 0]
    P0 = VPos[ITris[:, 0], :]
    P1 = VPos[ITris[:, 1], :]
    P2 = VPos[ITris[:, 2], :]

    #Compute normals
    FNormals = FNormals/FAreas[:, None]
    FAreas = 0.5*FAreas
    FNormals = FNormals
    VNormals = np.zeros_like(VPos)
    for k in range(3):
        VNormals[ITris[:, k], :] += FAreas[:, None]*FNormals
    mags = np.sqrt(np.sum(VNormals**2, axis=1))
    mags[mags <= 0] = 1
    return VNormals/mags[:, None]

def get_greedy_perm(X, idxs_start=[], n_perm=None):
    """
    Compute a furthest point sampling permutation of a set of points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data
    idxs_start: list of int
        The indices that have already been chosen
    n_perm: int
        Number of points to take in the permutation
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lam: float
        Covering radius
    """
    N = X.shape[0]
    if not n_perm:
        n_perm = X.shape[0]
    if len(idxs_start) == 0:
        # If no points have been chosen yet, use the point at index 0 to start
        idxs_start = [0]
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    idx_perm[0:len(idxs_start)] = np.array(idxs_start, dtype=int)
    dpoint2all = lambda i: np.sqrt(np.sum((X[i, :]-X)**2, axis=1))
    ds = np.inf*np.ones(N)
    for idx in idxs_start:
        ds = np.minimum(ds, dpoint2all(idx))
    for i in range(len(idxs_start), n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        ds = np.minimum(ds, dpoint2all(idx))
    lam = np.max(ds)
    return idx_perm, lam



"""#####################################
        Labeling Utilities
#####################################"""

def label_mesh_components(N, ITris, cluster_cutoff=0):
    """
    Label connected components of a mesh
    Parameters
    ----------
    N: int
        Number of vertices
    ITris: ndarray(T, 3, dtype=int)
        Triangle indices
    cluster_cutoff: int
        The minimum cluster size to consider

    Returns
    -------
    labels: ndarray(N, dtype=int)
        Connected component labels.  -1 if it was not part of 
        a connected component of cardinality >= cluster_cutoff
    """
    labels = -1*np.ones(N, dtype=int)
    neighbs = get_vertex_neighbors(ITris)

    ## Step 3: Run DFS from each node to find clusters
    visited = np.zeros(N)
    touched = np.zeros(N)
    label_idx = 0
    ## Loop through all vertices
    for i in range(N):
        if visited[i] == 0:
            ## We've found the beginning of a new cluster
            ## Start a depth-first search and add everything that is visited
            ## in the DFS to a tooth
            cluster = []
            stack = [i]
            while len(stack) > 0: # O(V) iterations
                i2 = stack.pop()
                cluster.append(i2)
                visited[i2] = 1
                for i3 in neighbs[i2]: # O(E) iterations
                    if touched[i3] == 0:
                        touched[i3] = 1
                        stack.append(i3)
            if len(cluster) > cluster_cutoff:
                cluster = np.array(cluster, dtype=int)
                labels[cluster] = label_idx
                label_idx += 1
    return labels


def label_volume_components(B, cluster_cutoff=0):
    """
    Label connected components of a binary volume
    Parameters
    ----------
    B: ndarray(M, N, L)
        A volume of 1s and 0s
    cluster_cutoff: int
        The minimum cluster size to consider

    Returns
    -------
    labels: ndarray(M, N, L, dtype=int)
        Connected component labels.  -1 if it was not part of 
        a connected component of cardinality >= cluster_cutoff
    """
    labels = -1*np.ones(B.shape, dtype=int)
    (M, N, L) = labels.shape

    ## Step 1: Run DFS from each node to find clusters
    visited = np.zeros(B.shape)
    touched = np.zeros(B.shape)
    label_idx = 0
    ## Loop through all vertices
    d_neighbs = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    d_neighbs = np.array(d_neighbs, dtype=int)
    for i in range(M):
        for j in range(N):
            for k in range(L):
                if B[i, j, k] == 1 and visited[i, j, k] == 0:
                    ## We've found the beginning of a new cluster
                    ## Start a depth-first search and add everything that is visited
                    ## in the DFS to a tooth
                    cluster = []
                    stack = [np.array([i, j, k], dtype=int)]
                    while len(stack) > 0: # O(V) iterations
                        ijk2 = stack.pop()
                        [i2, j2, k2] = ijk2
                        cluster.append(ijk2)
                        visited[i2, j2, k2] = 1
                        for dn in d_neighbs: # O(E) iterations
                            ijk3 = ijk2 + dn
                            [i3, j3, k3] = ijk3
                            if i3 >= 0 and i3 < M and j3 >= 0 and j3 < N and k3 >= 0 and k3 < L:
                                if B[i3, j3, k3] == 1 and touched[i3, j3, k3] == 0:
                                    touched[i3, j3, k3] = 1
                                    stack.append(ijk3)
                    if len(cluster) > cluster_cutoff:
                        cluster = np.array(cluster, dtype=int)
                        labels[cluster[:, 0], cluster[:, 1], cluster[:, 2]] = label_idx
                        label_idx += 1
    return labels

def get_label_counts(labels):
    """
    Parameters
    ----------
    labels: ndarray(N, dtype=int)
        Connected component labels.  -1 if it was not part of 
        a connected component of cardinality >= cluster_cutoff

    Returns
    -------
        ndarray(num_labels)
            Index i is the counts with label i
    """
    from scipy.sparse import coo_matrix
    labels = labels[labels >= 0]
    N = labels.size
    n_labels = len(np.unique(labels))
    counts = coo_matrix((np.ones(N), (np.zeros(N), labels)), shape=(1, n_labels))
    return np.array(counts.toarray().flatten(), dtype=int)

def crop_binary_volume(B):
    """
    Crop a binary volume to the axis-aligned bounding box containing
    all of the 1's

    Parameters
    ----------
    B: ndarray(M, N, L)
        A volume of 1s and 0s
    
    Returns:
    ndarray(<=M, <=N, <=L)
    """
    rg = []
    for i in range(3):
        Bi = B.swapaxes(0, i)
        Bi = np.reshape(Bi, (Bi.shape[0], Bi.shape[1]*Bi.shape[2]))
        idx = np.where(np.sum(Bi, axis=1) > 0)[0]
        if idx.size > 0:
            rg.append((np.min(idx), np.max(idx)))
    ret = np.array([])
    if len(rg) > 0:
        ret = B[rg[0][0]:rg[0][1], rg[1][0]:rg[1][1], rg[2][0]:rg[2][1]]
    return ret

def binary_volume_2coords(B):
    """
    Extract a point cloud representation of the voxels
    that are occupied in a binary volume
    
    Parameters
    ----------
    V: ndarray(M, N, L)
        A volume of 1s and 0s
    """
    return np.array(np.where(B==1), dtype=float).T