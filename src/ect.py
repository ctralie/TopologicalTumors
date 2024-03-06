import numpy as np

def get_wect_complex_img2d(img):
    """
    Compute the weighted Euler characteristic transform complex
    of a 2D image
    NOTE: The function values are not what a valid filtration would be,
    but some minor modifications to the code could make that so
    
    Parameters
    ----------
    img: ndarray(M, N)
        Grayscale image
        
    Returns
    -------
    {
        vertices:{
            v: ndarray(Nv, 2)
                Positions of vertices
            Vw: ndarray(Nv)
                Weight of vertices
        },
        
        faces: {
            idxs: ndarray(Nt, 3, dtype=int)
                Indices of vertices making up triangle
            Fw: ndarray(Nt)
                Weight of each face
        },
        
        edges: {
            idxs: ndarray(Ne, 2, dtype=int)
                Indices of vertices making up edge
            Ew: ndarray(Ne)
                Weight of each edge
        }
    
    }
    """
    res = {}
    ## Step 1: Setup vertices
    I = np.zeros((img.shape[0]*2+1, img.shape[1]*2+1))
    I[1:-1:2, 1:-1:2] = img > 0
    I[0:-1:2, 0:-1:2] += img > 0 
    I[2::2, 2::2] += img > 0 
    I[0:-1:2, 2::2] += img > 0
    I[2::2, 0:-1:2] += img > 0
    I[I > 1] = 1

    vals = np.zeros((img.shape[0]*2+1, img.shape[1]*2+1))
    vals[1:-1:2, 1:-1:2] = img
    vals[0:-1:2, 0:-1:2] = img 
    vals[2::2, 2::2] = np.maximum(vals[2::2, 2::2], img)
    vals[0:-1:2, 2::2] = np.maximum(vals[0:-1:2, 2::2], img)
    vals[2::2, 0:-1:2] = np.maximum(vals[2::2, 0:-1:2], img)

    ii, jj = np.meshgrid(np.arange(I.shape[0]), np.arange(I.shape[1]), indexing='ij')
    VIdxs = np.zeros(ii.shape, dtype=int)
    ii = ii[I > 0]
    jj = jj[I > 0]
    v = np.array([ii, jj]).T
    Vw = vals[I > 0]
    VIdxs[ii, jj] = np.arange(ii.size)
    
    # Normalize to the unit circle
    mu = np.sum(v*Vw[:, None], axis=0, keepdims=True)/np.sum(Vw)
    scale = np.max(np.sqrt(np.sum((v-mu)**2, axis=1))) ## TODO: A more stable way to scale?
    res["vertices"] = dict(v=np.fliplr(np.array([[-1, 1]])*(v-mu)/scale), Vw=Vw)

    ## Step 2: Setup triangles
    ii, jj = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    t = np.array([ii.flatten()*2+1, jj.flatten()*2+1]).T
    # Left, bottom, right, top
    t1 = np.concatenate((t, t, t, t), axis=0)
    t2 = np.concatenate((t+np.array([[1, -1]]), 
                         t+np.array([[-1, -1]]), 
                         t+np.array([[-1, 1]]), 
                         t+np.array([[1, 1]])), axis=0)
    t3 = np.concatenate((t+np.array([[-1, -1]]), 
                         t+np.array([[-1, 1]]), 
                         t+np.array([[1, 1]]), 
                         t+np.array([[1, -1]])), axis=0)
    vld = I[t1[:, 0], t1[:, 1]]*I[t2[:, 0], t2[:, 1]]*I[t3[:, 0], t3[:, 1]] > 0
    idxs = np.array([VIdxs[t1[vld, 0], t1[vld, 1]], 
                     VIdxs[t2[vld, 0], t2[vld, 1]], 
                     VIdxs[t3[vld, 0], t3[vld, 1]]], dtype=int).T
    res["tris"] = dict(idxs=idxs, Fw=img[t1[vld, 0]//2, t1[vld, 1]//2])

    ## Step 3: Setup edges
    ii, jj = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    ## Step 3a: Setup diagonal edges
    e = np.array([ii.flatten()*2+1, jj.flatten()*2+1]).T
    e1 = np.concatenate((e, e, e, e), axis=0)
    e2 = np.concatenate((e+np.array([[1, -1]]), 
                         e+np.array([[-1, -1]]), 
                         e+np.array([[-1, 1]]), 
                         e+np.array([[1, 1]])), axis=0)
    vld = I[e1[:, 0], e1[:, 1]]*I[e2[:, 0], e2[:, 1]] > 0
    idxs = np.array([VIdxs[e1[vld, 0], e1[vld, 1]], 
                     VIdxs[e2[vld, 0], e2[vld, 1]]], dtype=int).T
    Ew = img[e1[vld, 0]//2, e1[vld, 1]//2] # Edge Weights
    ## Step 3b: Setup horizontal edges
    e = np.array([ii.flatten()*2, jj.flatten()*2]).T
    e1b = np.array(e)
    e2b = e + np.array([[0, 2]])
    Ewb = img[e1b[:, 0]//2, e1b[:, 1]//2]
    idx = np.arange(e1b.shape[0])
    idx = idx[e1b[:, 0] > 0]
    Ewb[idx] = np.maximum(Ewb[idx], img[e1b[idx, 0]//2-1, e1b[idx, 1]//2])
    vld = I[e1b[:, 0]+1, e1b[:, 1]+1] + I[e1b[:, 0]-1, e1b[:, 1]+1] > 0
    idxs = np.concatenate((idxs,
                          np.array([VIdxs[e1b[vld, 0], e1b[vld, 1]], 
                                    VIdxs[e2b[vld, 0], e2b[vld, 1]]], dtype=int).T
                          ))
    Ew = np.concatenate((Ew, Ewb[vld]))
    ## Step 3c: Setup vertical edges
    e = np.array([ii.flatten()*2, jj.flatten()*2]).T
    e1b = np.array(e)
    e2b = e + np.array([[2, 0]])
    Ewb = img[e1b[:, 0]//2, e1b[:, 1]//2]
    idx = np.arange(e1b.shape[0])
    idx = idx[e1b[:, 1] > 0]
    Ewb[idx] = np.maximum(Ewb[idx], img[e1b[idx, 0]//2, e1b[idx, 1]//2-1])
    vld = I[e1b[:, 0]+1, e1b[:, 1]+1] + I[e1b[:, 0]+1, e1b[:, 1]-1] > 0
    idxs = np.concatenate((idxs,
                          np.array([VIdxs[e1b[vld, 0], e1b[vld, 1]], 
                                    VIdxs[e2b[vld, 0], e2b[vld, 1]]], dtype=int).T
                          ))
    Ew = np.concatenate((Ew, Ewb[vld]))
    res["edges"] = dict(idxs=idxs, Ew=Ew)
    
    return res
    

def get_curve(wect_complex, cres, dres, weighted=True):
    """
    Parameters
    ----------
    wect_complex:
        The weighted Euler characteristic transform complex.
        Coordinates are assumed to be inside of the unit circle
    cres: int
        Number of angles on the circle to take
    dres: int
        Resolution of the sublevelset filter between [-1, 1]
    weighted: bool
        If True, use weights.  If false, do unweighted
    """
    v = wect_complex["vertices"]["v"]
    Vw = wect_complex["vertices"]["Vw"]
    Ew = wect_complex["edges"]["Ew"]
    Fw = wect_complex["tris"]["Fw"]
    if not weighted:
        Vw = np.ones_like(Vw)
        Ew = np.ones_like(Ew)
        Fw = np.ones_like(Fw)
    
    t = np.linspace(0, 2*np.pi, cres+1)[0:cres]
    u = np.array([np.cos(t), np.sin(t)]).T
    
    # Row: direction, column: simplex
    pv = u.dot(v.T) 
    pe = np.inf*np.ones((u.shape[0], Ew.size))
    for i in range(2):
        pe = np.minimum(pe, pv[:, wect_complex["edges"]["idxs"][:, i]])
    pf = np.inf*np.ones((u.shape[0], Fw.size))
    for i in range(3):
        pf = np.minimum(pf, pv[:, wect_complex["tris"]["idxs"][:, i]])
    
    ds = np.linspace(-1, 1, dres)
    # 0: Direction, 1: weighted simplex, Column: distance
    chi =  np.sum(Vw[None, :, None]*(pv[:, :, None] >= ds[None, None, :]), axis=1)
    chi -= np.sum(Ew[None, :, None]*(pe[:, :, None] >= ds[None, None, :]), axis=1)
    chi += np.sum(Fw[None, :, None]*(pf[:, :, None] >= ds[None, None, :]), axis=1)
    
    return t, ds, chi
