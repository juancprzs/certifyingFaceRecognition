import sys
import scipy
import torch
import random
import numpy as np
from glob import glob
import os.path as osp
from collections import OrderedDict
from scipy.optimize import root_scalar
from scipy.spatial.distance import pdist

DEMO = False
BOUNDARIES_DIR = 'boundaries'
DATASETS = ['ffhq', 'celebahq']
GAN_NAMES = ['stylegan', 'pggan']
ATTRS = OrderedDict()
ATTRS['age'] = 0.5
ATTRS['eyeglasses'] = 0.5
ATTRS['gender'] = 0.2
ATTRS['pose'] = 0.5
ATTRS['smile'] = 0.8

# For deterministic behavior
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def sq_distance(A, shifted1, shifted2=None):
    if shifted2 is None:
        shifted2 = shifted1 
    if isinstance(A, torch.Tensor):
        transp = shifted1.permute(0, 2, 1).contiguous()
        temp = torch.bmm(A.repeat(shifted1.size(0), 1, 1), shifted2)
        result = torch.bmm(transp, temp)
    else:
        transp = np.transpose(shifted1, (0, 2, 1))
        temp = np.matmul(A, shifted2)
        result = np.matmul(transp, temp)
    
    return result.reshape(-1)


def sq_distance_diag(mat, vec1, vec2=None):
    # This function assumes mat is a vector representing a diagonal matrix
    if vec2 is None:
        prod = vec1**2
    else:
        prod = vec1 * vec2
    # Expand matrix to match dimensions in the batch size
    mat_size = mat.size(0)
    bs = vec1.size(0)
    mat = mat.view(1, 1, mat_size).expand(bs, 1, mat_size)
    dists = torch.bmm(mat, prod)
    return dists.reshape(-1)


def proj_ellipse(y, A, mu=None, c=1):
    """ 
        A: matrix defining ellipse (precision mat, i.e. inverse of cov mat)
            -- ellipse: X.T @ A @ X = 1
        mu : mean vector
        c: level set
        y: vector to be projected

        Example:
        points = np.random.uniform(low=-2, high=2, size=(SAMPLES, 2, 1)) + mu
        projs, _, _ = proj_ellipse(y, A)
    """
    def solve(A, shifted):
        def fun(t, vec):
            inv = np.linalg.inv(np.eye(A.shape[0]) + t*A)
            inter = np.matmul(inv, np.matmul(A, inv))
            return sq_distance(inter, vec) - 1
        
        bracket = [np.finfo(float).eps, 1e3]
        solutions = []
        for idx in range(shifted.shape[0]):
            f = lambda t: fun(t, shifted[idx].reshape(1, -1, 1))
            solution = root_scalar(f, method='bisect', bracket=bracket)
            solutions.append(solution.root)
        
        return np.array(solutions).reshape(-1, 1)

    if mu is None:
        mu = np.zeros((y.shape[0], 1))
    # Assertions
    assert len(mu.shape) == 2 and mu.shape[1] == 1, 'mu must be column vector'
    assert y.shape[-2] == mu.shape[-2], \
        'y should be a stack of vectors of the same dim as mu'

    # Reshapes
    mu = mu.reshape(1, -1, 1) # Add batch dim
    if len(y.shape) == 2: # Conditionally add batch dim
        n_dims, n_vecs = y.shape
        y = y.T.reshape(n_vecs, n_dims, 1)
    
    # Scale matrix and shift vector
    sc_A = A / c
    shifted = y - mu

    # Check y that are outside region and extract them, then project those
    sq_dist = sq_distance(sc_A, shifted)
    which_out = sq_dist > 1
    extracted = shifted[which_out]

    # Find scalar t's
    t = solve(sc_A, extracted)

    # Project with the t's that were found
    mat = np.expand_dims(t, 2) * np.expand_dims(sc_A, 0)
    mat = mat + np.expand_dims(np.eye(sc_A.shape[0]), 0)
    projections = np.linalg.solve(mat, extracted)
    shifted[which_out] = projections

    # Compute new distances
    sq_dist = sq_distance(sc_A, shifted)
    result = shifted + mu
    if result.shape[0] == 1: # Has batch dim == 1
        result = np.squeeze(result, axis=0)
    else: # Has batch dim > 1
        result = np.squeeze(result, axis=2).T
    
    return result, t, sq_dist <= 1


def proj_ellipse_pytorch(y, A, mu=None, c=1):
    """ 
        A: matrix defining ellipse (precision mat, i.e. inverse of cov mat)
            -- ellipse: X.T @ A @ X = 1
        mu : mean vector
        c: level set
        y: vector to be projected

        Example:
        points = np.random.uniform(low=-2, high=2, size=(SAMPLES, 2, 1)) + mu
        projs, _, _ = proj_ellipse(y, A)
    """
    def solve(A, shifted):
        def fun(t, vec):
            inv = torch.linalg.inv(torch.eye(A.shape[0], device=A.device) + t*A)
            # inter = torch.matmul(inv, torch.matmul(A, inv))
            inter = torch.linalg.multi_dot([inv, A, inv])
            return sq_distance(inter, vec) - 1
        
        bracket = [np.finfo(float).eps, 1e3]
        sols, which_out = [], []
        for idx in range(shifted.shape[0]):
            f = lambda t: fun(t, shifted[idx].reshape(1, -1, 1))
            eval1, eval2 = f(bracket[0]).item(), f(bracket[1]).item()
            if eval1 * eval2 < 0: # Opposing signs
                solution = root_scalar(f, method='bisect', bracket=bracket)
                sols.append(solution.root)
                which_out.append(True)
            else:
                which_out.append(False)
        
        return (torch.tensor(sols, device=A.device).reshape(-1, 1), 
            torch.tensor(which_out, device=A.device))

    if mu is None:
        mu = torch.zeros((y.shape[0], 1), device=y.device)
    # Assertions
    assert len(mu.shape) == 2 and mu.shape[1] == 1, 'mu must be column vector'
    assert y.shape[-2] == mu.shape[-2], \
        'y should be a stack of vectors of the same dim as mu'

    # Reshapes
    mu = mu.reshape(1, -1, 1) # Add batch dim
    if len(y.shape) == 2: # Conditionally add batch dim
        n_dims, n_vecs = y.shape
        y = y.T.reshape(n_vecs, n_dims, 1)
    
    # Scale matrix and shift vector
    sc_A = A / c
    shifted = y - mu

    # Check y that are outside region and extract them, then project those
    # sq_dist = sq_distance(sc_A, shifted)
    # which_out = sq_dist > 1
    # extracted = shifted[which_out]

    # Find scalar t's
    # t = solve(sc_A, extracted)
    t, which_out = solve(sc_A, shifted)
    extracted = shifted[which_out]

    # Project with the t's that were found
    mat = torch.unsqueeze(t, 2) * torch.unsqueeze(sc_A, 0)
    mat = mat + torch.unsqueeze(torch.eye(sc_A.shape[0], device=A.device), 0)
    projections = torch.linalg.solve(mat, extracted)
    shifted[which_out] = projections

    # Compute new distances
    sq_dist = sq_distance(sc_A, shifted)
    result = shifted + mu
    if result.shape[0] == 1: # Has batch dim == 1
        result = torch.squeeze(result, dim=0)
    else: # Has batch dim > 1
        result = torch.squeeze(result, dim=2).T
    
    return result, t, sq_dist <= 1


def proj_ellipse_pytorch_diag(y, vec_A, mu=None, c=1):
    """ 
        vec_A: vector modeling the DIAGONAL matrix that defined the ellipse
            -- ellipse: X.T @ A @ X = 1
        mu : mean vector
        c: level set
        y: vector to be projected

        Example:
        points = np.random.uniform(low=-2, high=2, size=(SAMPLES, 2, 1)) + mu
        projs, _, _ = proj_ellipse(y, A)
    """
    def solve(vec_A, shifted):
        def fun(t, vec):
            # Equivalent to B = (I + t*A)^(-1)
            inv = 1 / (1 + t*vec_A)
            # Equivalent to C = B @ A @ B
            prod = vec_A * inv**2
            # Equivalent to vec.T @ C @ vec (but without summing)
            inter = prod * vec.squeeze()**2
            return inter.sum() - 1
        
        bracket = [np.finfo(float).eps, 1e3]
        sols, which_out = [], []
        for idx in range(shifted.shape[0]):
            f = lambda t: fun(t, shifted[idx].reshape(1, -1, 1))
            eval1, eval2 = f(bracket[0]).item(), f(bracket[1]).item()
            if eval1 * eval2 < 0: # Opposing signs
                solution = root_scalar(f, method='bisect', bracket=bracket)
                sols.append(solution.root)
                which_out.append(True)
            else:
                which_out.append(False)
        
        return (torch.tensor(sols, device=y.device).reshape(-1, 1), 
            torch.tensor(which_out, device=y.device))

    if mu is None:
        mu = torch.zeros((y.shape[0], 1), device=y.device)
    # Assertions
    assert len(mu.shape) == 2 and mu.shape[1] == 1, 'mu must be column vector'
    assert y.shape[-2] == mu.shape[-2], \
        'y should be a stack of vectors of the same dim as mu'

    # Reshapes
    mu = mu.reshape(1, -1, 1) # Add batch dim
    if len(y.shape) == 2: # Conditionally add batch dim
        n_dims, n_vecs = y.shape
        y = y.T.reshape(n_vecs, n_dims, 1)
    
    # Scale matrix and shift vector
    sc_vec_A = vec_A / c
    shifted = y - mu

    # Find scalar t's
    t, which_out = solve(sc_vec_A, shifted)
    extracted = shifted[which_out]

    # Solve system with the t's that were found
    # System can be solved efficiently by inverting because the matrix is diag
    mat = 1 + t * sc_vec_A.unsqueeze(0)
    inv = 1 / mat
    projections = inv.unsqueeze(2) * extracted
    shifted[which_out] = projections

    # Compute new distances
    sq_dist = sq_distance(torch.diag(sc_vec_A), shifted)
    result = shifted + mu
    if result.shape[0] == 1: # Has batch dim == 1
        result = torch.squeeze(result, dim=0)
    else: # Has batch dim > 1
        result = torch.squeeze(result, dim=2).T
    
    return result, t, sq_dist <= 1


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_full_points(points, fill_with_null=False, get_exp_out=True):
    if fill_with_null:
        # Compute nullspace
        null = scipy.linalg.null_space(points.T)
        points = np.concatenate([points, null], axis=1)
        assert points.shape[0] == points.shape[1]
    
    # Get the complementary points and the expected output
    # Concatenate with -points (mirrored version of points)
    full_points = np.concatenate((points, -points), axis=1)
    # points.shape == (dim, N)
    tot_N_points = full_points.shape[1]
    if get_exp_out:
        exp_out = np.zeros((tot_N_points, tot_N_points))
        orig_N_points = points.shape[1]
        for idx in range(orig_N_points):
            exp_out[idx, idx+orig_N_points] = -1

        exp_out = exp_out + exp_out.T + np.eye(tot_N_points)
    else:
        exp_out = None
    return full_points, exp_out


def my_mvee(points, exp_out):
    # P, exp_out = get_full_points(P, fill_with_null=True)
    # # can even work with fill_with_null=False
    # my_X = my_mvee(P, exp_out) # My way
    # X = modify_mat(my_X)
    # if not check_mat(X, P, exp_out):
    #     print('Problem with modified matrix')
    #     sys.exit()
    P_inv = np.linalg.pinv(points) # right pseudoinverse: I = P @ P_inv
    return P_inv.T @ exp_out @ P_inv


def ellipse(u, v, rx, ry, rz):
    x = rx * np.cos(u) * np.cos(v)
    y = ry * np.sin(u) * np.cos(v)
    z = rz * np.sin(v)
    return x, y, z


def get_xyz_ellipse(mat):
    U, D, V = np.linalg.svd(mat)
    rx, ry, rz = 1. / np.sqrt(D)
    u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]
    E = np.dstack(ellipse(u, v, rx, ry, rz))
    E = np.dot(E, V)
    x, y, z = np.rollaxis(E, axis=-1)
    return x, y, z


def modify_mat(mat):
    vals, vecs = np.linalg.eigh(mat) # Matrix is Hermitian
    # Modify eigenvalues
    where = vals < 1e-3
    # where = np.ones_like(vals, dtype=bool)
    vals[where] = vals.max()
    # Reconstructing the matrix so that eigenvectors are preserved
    new_mat = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
    return new_mat


def check_mat(mat, P, exp_out):
    # Matrix should be symmetric
    if not np.allclose(mat, mat.T):
        return False
    eig_vals = np.round(np.linalg.eigvalsh(mat), 4)
    # Matrix should be PD
    if np.any(eig_vals < 0):
        return False
    # Points should be on the ellipse
    if not np.allclose(P.T @ mat @ P, exp_out):
        return False
    
    return True


def sample_ellipsoid(ellipsoid_mat, n_vecs=1):
    # See Wikipedia:
    # https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_within_the_n-ball
    n = ellipsoid_mat.shape[0]
    # # Sample from unit n-ball
    # 'vec' will be uniformly distributed over the surface of the n-ball
    # i.e. uniformly over the (n-1)-sphere
    # From normal distribution and normalize
    if isinstance(ellipsoid_mat, torch.Tensor):
        vec = torch.randn(n, n_vecs, device=ellipsoid_mat.device)
        vec /= torch.norm(vec, dim=0)
        # 'vec' provides uniformly distributed directions -> Now need radius
        # Sample uniform radius and then scale according to dimension of ball
        rad = torch.rand(n_vecs, device=ellipsoid_mat.device) ** (1/n)
        vec *= rad
        # Deform to get ellipsoid
        # Get Cholesky decomp: np.allclose(chol @ chol.T, ellipsoid_mat) is True
        chol = torch.linalg.cholesky(ellipsoid_mat)
        transform = torch.linalg.inv(chol.T) # To map from ball to ellipse
    else:
        vec = np.random.randn(n, n_vecs)
        vec /= np.linalg.norm(vec, axis=0)
        # 'vec' provides uniformly distributed directions -> Now need radius
        # Sample uniform radius and then scale according to dimension of ball
        rad = np.random.rand(n_vecs) ** (1/n)
        vec *= rad
        # Deform to get ellipsoid
        # Get Cholesky decomp: np.allclose(chol @ chol.T, ellipsoid_mat) is True
        chol = np.linalg.cholesky(ellipsoid_mat)
        transform = np.linalg.inv(chol.T) # To map from ball to ellipse
    
    transformed = transform @ vec # Map them
    return transformed.T


def mvee(points, tol=0.001):
    # Taken from 
    # https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    # https://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
    # https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * np.linalg.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0) / ((d+1) * (M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    
    c = u * points
    A = np.linalg.inv(points.T * np.diag(u) * points - c.T * c) / d    
    return np.asarray(A), np.squeeze(np.asarray(c))


def project_to_region(vs, proj_mat, ellipse_mat, check=True, dirs=None, 
        on_surface=False):
    '''
    vs: vectors to project (d x n1)
    proj_mat: matrix to project vectors to subspace spanned by directions (cols 
        of 'dirs')
    ellipse_mat: matrix parameterizing the hyper-ellipsoid (d x d)
    check: boolean stating whether projections are to be checked
    dirs: matrix of vectors establishing directions of interest (d x n2)
    '''
    # Project to space spanned by dirs
    vs = vs.T
    proj_subs = proj_mat @ vs # Project vector to subspace
    if on_surface:
        dists = sq_distance(ellipse_mat, np.expand_dims(proj_subs.T, axis=2))
        proj_ell = proj_subs / np.sqrt(dists)
    else:
        # The projection of query vector onto the ellipse
        proj_ell, _, _ = proj_ellipse(proj_subs, ellipse_mat)

    if check:
        assert dirs is not None, \
            "Must provide 'dirs' if want to check projection"
        # Check for subspace
        proj_subs2 = dirs @ np.linalg.pinv(dirs) @ vs
        assert np.allclose(proj_subs, proj_subs2), 'Projection to subspace is '\
            'wrong'
        assert np.allclose(proj_mat @ proj_ell, proj_ell), 'Points inside '\
            'ellipse should also be on the subspace'
        # Check for ellipse
        ellps_dist = sq_distance(
            ellipse_mat, np.expand_dims(proj_ell.T, axis=2)
        )
        assert np.allclose(ellps_dist[ellps_dist > 1.], 1), \
            'Some points outside ellipsoid!'
    
    return proj_ell.T, proj_subs.T


def in_subs(v, proj_mat):
    dists2subs = torch.norm(proj_mat @ v - v, p=2, dim=0)
    whch_out = dists2subs > 0.
    return torch.allclose(dists2subs[whch_out], torch.tensor(0.), atol=1e-4)


def in_ellps(v, ellipse_mat, atol=1e-4):
    dists = sq_distance(ellipse_mat, v.T.unsqueeze(dim=2))
    whch_out = dists > 1.
    return torch.allclose(dists[whch_out], torch.tensor(1.), atol=atol)


def proj2region(vs, proj_mat, ellipse_mat, check=True, dirs=None, to_subs=True,
        on_surface=False, max_iters=5, diag_ellipse_mat=False):
    '''
    vs: vectors to project (d x n1)
    proj_mat: matrix to project vectors to subspace spanned by directions (cols 
        of 'dirs')
    ellipse_mat: matrix parameterizing the hyper-ellipsoid (d x d)
    check: boolean stating whether projections are to be checked
    dirs: matrix of vectors establishing directions of interest (d x n2)
    diag_ellipse_mat: boolean stating if ellipse_mat is diagonal -> improve 
        running time
    '''
    ell_mat = torch.diag(ellipse_mat) if diag_ellipse_mat else ellipse_mat
    def proj2surf(v):
        dists = sq_distance(ell_mat, v.T.unsqueeze(dim=2))
        sqrt_dists = torch.sqrt(dists.reshape(1, -1))
        return v / (sqrt_dists + 1e-4)

    def condition(proj):
        if to_subs:
            return in_subs(proj, proj_mat) and in_ellps(proj, ell_mat)
        else:
            return in_ellps(proj, ell_mat)
    
    vs = vs.T
    # Project vector to subspace (as spanned by dirs)
    proj_subs = proj_mat @ vs if to_subs else vs

    # Projection ON the ellipse (if requested)
    if on_surface: proj_subs = proj2surf(proj_subs)

    # Ellipse-projection function    
    if diag_ellipse_mat: # Much more efficient
        proj_ellipse_fun = proj_ellipse_pytorch_diag
    else:
        proj_ellipse_fun = proj_ellipse_pytorch
    
    # Projection INSIDE the ellipse
    proj_ell, _, _ = proj_ellipse_fun(proj_subs, ellipse_mat)

    # Iterate until max_iters if needed
    curr_iters = 0
    while not condition(proj_ell):
        if curr_iters == max_iters: break
        curr_iters += 1
        # The projection of query vector onto the ellipse
        proj_ell, _, _ = proj_ellipse_fun(proj_ell, ellipse_mat)
        # Project vector to subspace
        proj_ell = proj_mat @ proj_ell if to_subs else proj_ell
    
    # Final projection for vectors that are still a bit outside
    if not condition(proj_ell):
        dists = sq_distance(ell_mat, proj_ell.T.unsqueeze(dim=2))
        whr_need = (torch.sqrt(dists.reshape(1, -1)) >= 1).squeeze()
        proj_ell[:, whr_need] = proj2surf(proj_ell)[:, whr_need]

    if check:
        if to_subs:
            assert dirs is not None, "Must provide 'dirs' if want to check " \
                "projection"
            # Check for subspace
            assert torch.allclose(dirs @ torch.linalg.pinv(dirs), proj_mat, 
                atol=5e-4), 'Projection matrix to subspace is wrong'
            assert in_subs(proj_ell, proj_mat), 'Points inside ellipse should '\
                'also be on the subspace'
        # Check for ellipse
        assert in_ellps(proj_ell, ell_mat), 'Some points outside ellipsoid!'
    
    return proj_ell.T, proj_subs.T


def get_normal_and_check(P, proj_ell, proj_pla, ellipse_mat):
    # Get the plane for plotting
    normal = np.cross(P[:, 0], P[:, 1])
    normal = normal / np.linalg.norm(normal)
    # Check things
    if np.allclose(proj_pla.T @ normal, 0):
        print('"plane-proj" is on the plane')
    else:
        print('Error: "plane-proj" is NOT on the plane!')

    if np.allclose(proj_ell.T @ normal, 0):
        print('"ellipse-proj" is on the plane')
    else:
        print('Error: "ellipse-proj" is NOT on the plane!')

    ellipse_dist = proj_ell.T @ ellipse_mat @ proj_ell 
    if (ellipse_dist < 1) or np.allclose(ellipse_dist, 1):
        print('"ellipse-proj" is inside the ellipse')
    else:
        print('Error: "ellipse-proj" is NOT inside the ellipse!')

    return normal


def get_original_points(random_points):
    # Points defining the ellipse (the directions of interest, in 
    # our case)
    if random_points:
        P = np.random.randn(256, 5)
    else:
        p1, p2 = np.array([3, 1, 2]), np.array([-1, 1, 0])
        p1, p2 = p1 / 2, p2 / 2
        P = np.concatenate([
            np.expand_dims(x, axis=1) for x in [p1, p2]], 
            axis=1
        )
    
    return P # P.shape == [n_dims, n_dirs]


def get_proj_mat(dirs):
    # 'dirs' is the matrix of directions of interest 
    # dirs.shape == [n_dims, n_dirs]
    return dirs @ np.linalg.pinv(dirs)


def plot_inner_prods(dirs):
    import matplotlib
    import matplotlib.pyplot as plt
    inn_prods = np.round(dirs.T @ dirs, 3)

    # Inspired by
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    im = ax.imshow(inn_prods)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(ATTRS))); ax.set_yticks(np.arange(len(ATTRS)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(list(ATTRS.keys()), fontsize=24)
    ax.set_yticklabels(list(ATTRS.keys()), fontsize=24)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ATTRS)):
        for j in range(len(ATTRS)):
            text = ax.text(j, i, inn_prods[i, j], ha="center", va="center", 
                           color="w", fontsize=18)

    ax.set_title("Inner products", fontsize=26)
    fig.tight_layout()
    plt.show()


def get_projection_matrices(dataset=DATASETS[0], gan_name=GAN_NAMES[0],
        attrs2drop=[], scale_factor=1.0):
    plot_mat = False
    file_template = osp.join(BOUNDARIES_DIR, 
                             f'{gan_name}_{dataset}_%s_w_boundary.npy')

    all_bounds = glob(osp.join(BOUNDARIES_DIR, '*.npy'))

    # Drop the unwanted attributes
    if len(attrs2drop) != 0:
        for attr in attrs2drop:
            assert attr in ATTRS.keys(), f'Attribute {attr} is NOT valid'
            ATTRS.pop(attr)

        print(f'Dropped {len(attrs2drop)} attributes -> {len(ATTRS)} remaining')

    # Load the directions
    dirs, files, magns = [], [], []
    for att_name, magn in ATTRS.items():
        this_file = file_template % att_name
        assert this_file in all_bounds, \
            f'Boundary for attr "{att_name}" not found!'
        # Load vector and modify its magnitude
        this_dir = np.load(this_file) # magn * np.load(this_file)
        magns.append(magn)
        dirs.append(this_dir)
        files.append(this_file)

    dirs = np.concatenate(dirs, axis=0).T
    # dirs.shape == [n_dims, n_dirs] == [n_dims, len(ATTRS)]
    assert dirs.shape[1] == len(ATTRS)

    if plot_mat:
        plot_inner_prods(dirs)

    proj_mat = get_proj_mat(dirs) # proj_mat.shape == [n_dims, n_dims]
    ellipse_mat = get_ellipse_mat(dirs)
    ellipse_mat = scale_factor * ellipse_mat

    # We also compute a lower-dimensional version of the ellipse matrix. This 
    # matrix represents a hyper-ellipsoid that DOES NOT live in the original
    # high-dimensional space, but on a space of dimension equivalent to the 
    # number of the direction of interest. The semi-axes of this ellipsoid are
    # aligned with the space canonical axis
    magns = np.array(magns)
    # Each magnitude is represented as axis-aligned vector
    red_dirs = np.diag(magns)
    red_ellipse_mat = get_ellipse_mat(red_dirs)
    red_ellipse_mat = scale_factor * red_ellipse_mat
    assert np.all(red_ellipse_mat == np.diag(np.diagonal(red_ellipse_mat))), \
        'Matrix should be diagonal'
    red_ellipse_mat = np.diagonal(red_ellipse_mat)

    # Return:
    # (1) proj to subspace, (2) mat parameterizing ellipse, (3) directions, 
    # (4) mat parameterizing reduced ellipse, and (5) files from which the 
    # directions came
    return proj_mat, ellipse_mat, dirs, red_ellipse_mat, files


def get_ellipse_mat(dirs):
    dirs_expanded, _ = get_full_points(dirs, fill_with_null=True)
    ellipse_mat, c = mvee(dirs_expanded.T) # Their way
    assert np.allclose(c, 0), "The origin should be the ellipses's center"
    # dirs_expanded, exp_out = get_full_points(dirs, fill_with_null=True)
    # my_X = my_mvee(dirs_expanded, exp_out)
    # assert np.allclose(X, my_X)
    return ellipse_mat


def transform_vecs(dirs):
    '''
    dirs is a matrix of shape [n_dims, n_vecs]
    '''
    norms = np.linalg.norm(dirs, axis=0)
    dot_prods = dirs.T @ dirs
    n_vecs = dirs.shape[1]
    new_dirs = np.zeros((n_vecs, n_vecs))
    new_dirs[0, 0] = norms[0]
    for idx in range(1, n_vecs):
        curr_dot_prods = dot_prods[idx, :idx]
        curr_mat = new_dirs[:idx, :idx]
        # Solve system to find first n=idx vector coords
        partial_vec = np.linalg.solve(curr_mat.T, curr_dot_prods)
        new_dirs[:idx, idx] = partial_vec 
        # Last coord is determined by norm
        inner_prod = partial_vec.T @ partial_vec
        last_coord = norms[idx]**2 - inner_prod
        new_dirs[idx, idx] = np.sqrt(last_coord)

    # Check dot products
    assert np.allclose(dot_prods, new_dirs.T @ new_dirs, atol=5e-4)
    # Check distances (Yes, I know this assertion is redundant)
    assert np.allclose(pdist(dirs.T), pdist(new_dirs.T))
    return new_dirs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    plot = False
    random_points = not plot
    compare_to_mine = True

    np.random.seed(111)
    P = get_original_points(random_points)

    # Projection matrix (to subspace spanned by the columns of P)
    proj_mat = get_proj_mat(P)

    # Get complementary points
    P_expanded, exp_out = get_full_points(P, fill_with_null=True)
    X, c = mvee(P_expanded.T) # Their way
    assert np.allclose(c, 0), "The origin should be the ellipses's center"

    if not check_mat(X, P_expanded, exp_out):
        print('Problem with computed matrix')
        sys.exit()

    if compare_to_mine:
        my_X = my_mvee(P_expanded, exp_out) # My way
        if np.allclose(X, my_X):
            print('Produces SAME result as my way')
        else:
            print('DOES NOT produce same result as my way')


    # Sample points inside ellipse
    ellipse_points = sample_ellipsoid(X, n_vecs=100)
    # Project points inside ellipse to subspace (plane)
    _, proj_subspace = project_to_region(ellipse_points, proj_mat, X, 
        check=True, dirs=P)
    proj_ellipse_points = proj_subspace
    # Check points are inside ellipse
    dists = sq_distance(X, np.expand_dims(proj_ellipse_points.T, axis=2))
    if not np.all(dists <= 1):
        print('Error: problem with sampled points!')
        sys.exit()


    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # The query vector for plotting
        q = np.random.randn(P.shape[0]) # np.array([0, 0, 6])
        q = q.reshape(-1, 1)
        # Proj is to a subspace -> in the case of visualization it's a plane!
        proj_ell, proj_pla = project_to_region(q, proj_mat, X, check=True, dirs=P)

        # Get the plane's normal vector for plotting
        normal = get_normal_and_check(P, proj_ell, proj_pla, X)
        # Check points are in plane
        if np.allclose(proj_ellipse_points.T @ normal, 0):
            print('"plane-proj samples" is on the plane')
        else:
            print('Error: "plane-proj" is NOT on the plane!')


        # Get the coordinates of the ellipse for plotting
        x_ell, y_ell, z_ell = get_xyz_ellipse(X)


        '''
                                    First plot       
        '''
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d'); ax.view_init(elev=25, azim=45)
        ax.scatter(0, 0, 0, label='origin') # The origin

        # The points
        ax.scatter(P_expanded[0, :2], P_expanded[1, :2], P_expanded[2, :2], 
            label='points')
        ax.scatter(P_expanded[0, 2:], P_expanded[1, 2:], P_expanded[2, 2:], 
            label='extra points')
        # The ellipsoid
        ellps = ax.plot_surface(x_ell, y_ell, z_ell, cstride=1, rstride=1, alpha=0.5, 
            label='ellipsoid', cmap='coolwarm')
        ellps._facecolors2d, ellps._edgecolors2d = ellps._facecolor3d, ellps._edgecolor3d

        # The plane
        set_axes_equal(ax)
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 501), np.linspace(y_lo, y_hi, 501))
        z = (-normal[0] * xx - normal[1] * yy - 0) * 1. / normal[2]
        plane = ax.plot_surface(xx, yy, z, alpha=0.3, label='plane', cmap='jet')
        plane._facecolors2d, plane._edgecolors2d = plane._facecolor3d, plane._edgecolor3d
        ax.legend()

        '''
                                    Second plot       
        '''
        ax = fig.add_subplot(222, projection='3d'); ax.view_init(elev=25, azim=45)
        ax.scatter(0, 0, 0, label='origin') # The origin

        # The ellipsoid
        ellps = ax.plot_surface(x_ell, y_ell, z_ell, cstride=1, rstride=1, alpha=0.5, 
            label='ellipsoid', cmap='coolwarm')
        ellps._facecolors2d, ellps._edgecolors2d = ellps._facecolor3d, ellps._edgecolor3d
        # Query vector
        ax.scatter(q[0, 0], q[1, 0], q[2, 0], color='orange', label='q')
        ax.scatter(proj_pla[0, 0], proj_pla[1, 0], proj_pla[2, 0], color='k', 
            label='plane-proj (p)') 
        # The projection of the vector onto the ellipse
        ax.scatter(proj_ell[0, 0], proj_ell[1, 0], proj_ell[2, 0], color='r', 
            label='ellipse-proj (r)') 
        # The plane
        set_axes_equal(ax)
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 501), np.linspace(y_lo, y_hi, 501))
        z = (-normal[0] * xx - normal[1] * yy - 0) * 1. / normal[2]
        plane = ax.plot_surface(xx, yy, z, alpha=0.3, label='plane', cmap='jet')
        plane._facecolors2d, plane._edgecolors2d = plane._facecolor3d, plane._edgecolor3d
        ax.legend()

        '''
                                    Third plot       
        '''
        ax = fig.add_subplot(223, projection='3d'); ax.view_init(elev=25, azim=45)
        ax.scatter(0, 0, 0, label='origin') # The origin
        ax.scatter(ellipse_points[0, :], ellipse_points[1, :], ellipse_points[2, :], 
            label='samples')
        # The ellipsoid
        ellps = ax.plot_surface(x_ell, y_ell, z_ell, cstride=1, rstride=1, alpha=0.5, 
            label='ellipsoid', cmap='coolwarm')
        ellps._facecolors2d, ellps._edgecolors2d = ellps._facecolor3d, ellps._edgecolor3d
        ax.legend()

        '''
                                    Fourth plot       
        '''
        ax = fig.add_subplot(224, projection='3d'); ax.view_init(elev=25, azim=45)
        ax.scatter(0, 0, 0, label='origin') # The origin
        ax.scatter(proj_ellipse_points[0, :], proj_ellipse_points[1, :], 
            proj_ellipse_points[2, :], label='plane proj samples')
        # The ellipsoid
        ellps = ax.plot_surface(x_ell, y_ell, z_ell, cstride=1, rstride=1, alpha=0.5, 
            label='ellipsoid', cmap='coolwarm')
        ellps._facecolors2d, ellps._edgecolors2d = ellps._facecolor3d, ellps._edgecolor3d
        # The plane
        set_axes_equal(ax)
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 501), np.linspace(y_lo, y_hi, 501))
        z = (-normal[0] * xx - normal[1] * yy - 0) * 1. / normal[2]
        plane = ax.plot_surface(xx, yy, z, alpha=0.3, label='plane', cmap='jet')
        plane._facecolors2d, plane._edgecolors2d = plane._facecolor3d, plane._edgecolor3d
        ax.legend()

        plt.show()

if __name__ == '__main__':
    proj_mat, ellipse_mat, dirs = get_projection_matrices()
    # With these matrices, we can project to region with
    # proj_reg, proj_subs = project_to_region(q, proj_mat, ellipse_mat, 
    #     check=True, dirs=dirs)
    if DEMO:
        main()
