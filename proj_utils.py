import sys
import scipy
import numpy as np
from glob import glob
import os.path as osp
from scipy.optimize import root_scalar

DEMO = False
BOUNDARIES_DIR = 'boundaries'
DATASET = ['ffhq', 'celebahq']
GAN_NAME = ['stylegan', 'pggan']
ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']

def sq_distance(A, shifted):
    transp = np.transpose(shifted, (0, 2, 1))
    result = np.squeeze(np.matmul(transp, np.matmul(A, shifted)))
    return result.reshape(-1)


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
    return transformed


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


def project_to_region(vs, proj_mat, ellipse_mat, check=True, dirs=None):
    '''
    vs: vectors to project (d x n1)
    proj_mat: matrix to project vectors to subspace spanned by directions (cols 
        of 'dirs')
    ellipse_mat: matrix parameterizing the hyper-ellipsoid (d x d)
    check: boolean stating whether projections are to be checked
    dirs: matrix of vectors establishing directions of interest (d x n2)
    '''
    # Project to space spanned by dirs
    proj_subs = proj_mat @ vs # Project vector to subspace
    # The projection of query vector onto the ellipse
    proj_ell, _, _ = proj_ellipse(proj_subs, ellipse_mat)

    if check:
        assert dirs is not None, \
            "Must provide 'dirs' if want to check projection"
        # Check for subspace
        proj_subs2 = dirs @ np.linalg.pinv(dirs) @ vs
        # Check for ellipse
        ellps_dist = sq_distance(ellipse_mat, np.expand_dims(proj_ell.T, axis=2))
        assert np.allclose(ellps_dist[ellps_dist > 1.], 1), \
            'Some points outside ellipsoid!'
    
    return proj_ell, proj_subs


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
    ax.set_xticklabels(ATTRS, fontsize=24)
    ax.set_yticklabels(ATTRS, fontsize=24)

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


def get_projection_matrices(dataset=DATASET[0], GAN_NAME=GAN_NAME[0]):
    plot_mat = False
    FILE_TEMPLATE = osp.join(BOUNDARIES_DIR, 
                             f'{GAN_NAME}_{dataset}_%s_w_boundary.npy')

    all_bounds = glob(osp.join(BOUNDARIES_DIR, '*.npy'))

    dirs = []
    for att in ATTRS:
        this_file = FILE_TEMPLATE % att
        assert this_file in all_bounds, f'Boundary for attr "{att}" not found!'
        dirs.append(np.load(this_file))

    dirs = np.concatenate(dirs, axis=0).T
    # dirs.shape == [n_dims, n_dirs] == [n_dims, len(ATTRS)]
    assert dirs.shape[1] == len(ATTRS)

    if plot_mat:
        plot_inner_prods(dirs)

    proj_mat = get_proj_mat(dirs) # proj_mat.shape == [n_dims, n_dims]
    dirs_expanded, _ = get_full_points(dirs, fill_with_null=True)
    ellipse_mat, c = mvee(dirs_expanded.T) # Their way
    assert np.allclose(c, 0), "The origin should be the ellipses's center"
    # dirs_expanded, exp_out = get_full_points(dirs, fill_with_null=True)
    # my_X = my_mvee(dirs_expanded, exp_out)
    # assert np.allclose(X, my_X)

    # Return
    # proj to subspace, mat parameterizing ellipse and directions
    return proj_mat, ellipse_mat, dirs


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
    import pdb; pdb.set_trace()
    if DEMO:
        main()
