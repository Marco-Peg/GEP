# import igl
import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse as sp


def Laplacian_eigen_decompostion(verts, faces=None, n_eigen=42):
    eps = 1e-8
    if faces is not None:
        # L = -igl.cotmatrix(verts, faces)
        # M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
        L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts, faces)
        massvec_np += eps * np.mean(massvec_np)
        M = sp.diags(massvec_np)
    else:
        L, M = robust_laplacian.point_cloud_laplacian(verts)
        # M = M.diagonal()

    try:
        evals, evecs = sp.linalg.eigsh(L, n_eigen, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs = sp.linalg.eigsh(L - 1e-8 * sp.identity(verts.shape[0]), n_eigen,
                                       M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    # Just for numerical stability
    evals[evals < 1e-6] = 1e-6
    return evals, evecs


def GPS(verts, faces=None, n_eigen=42):
    evals, evecs = Laplacian_eigen_decompostion(verts, faces, n_eigen)
    gps = - evecs / np.sqrt(evals).T

    return gps


def HKS(verts, faces=None, n_eigen=42, hks_size=100):
    # Number of vertices
    n = verts.shape[0]
    hks = np.zeros((n, hks_size))
    evals, evecs = Laplacian_eigen_decompostion(verts, faces, n_eigen)

    # hks_size samples logarithmically scaled
    t_min = 4 * np.log(10) / evals[-1]
    t_max = 4 * np.log(10) / evals[2]
    t = np.logspace(np.log10(t_min), np.log10(t_max), num=hks_size)

    for i in np.arange(0, hks_size):
        # Computing hks
        # print(t[i])
        hks[:, i] = np.sum(np.exp(-(evals.T * t[i])) * (evecs) ** 2, axis=1)

    return hks


def WKS(verts, faces=None, n_eigen=42, wks_size=100, variance=7):
    # Number of vertices
    n = verts.shape[0]
    WKS = np.zeros((n, wks_size))
    evals, evecs = Laplacian_eigen_decompostion(verts, faces, n_eigen)

    # log(E)
    log_E = np.log(evals).T
    # Define the energies step
    e = np.linspace(log_E[1], np.max(log_E) / 1.02, wks_size)
    # Compute the sigma
    sigma = (e[1] - e[0]) * variance
    C = np.zeros((wks_size, 1))

    for i in np.arange(0, wks_size):
        # Computing WKS
        WKS[:, i] = np.sum(
            (evecs) ** 2 * np.tile(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma ** 2)), (n, 1)), axis=1)
        # Noramlization
        C[i] = np.sum(np.exp((-(e[i] - log_E) ** 2) / (2 * sigma ** 2)))

    WKS = np.divide(WKS, np.tile(C, (1, n)).T)
    return WKS
