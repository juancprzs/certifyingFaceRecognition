import math

import torch
from torch.nn import functional as F

from scipy.linalg import null_space # Our modification


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()
    # Dot product here
    sign = 2 * ((w * t).sum(1) - b >= 0) - 1
    w.mul_(sign.unsqueeze(1))
    b.mul_(sign)

    a = (w < 0).float()
    d = (a - t) * (w != 0).float()

    p = a - t * (2 * a - 1)
    indp = torch.argsort(p, dim=1)
    # Dot products here
    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = indp.flip((1,))
    ws = w.gather(1, indp2)
    bs2 = - ws * d.gather(1, indp2)

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    b2 = sb[:, -1] - s[:, -1] * p.gather(1, indp[:, 0:1]).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)

        counter2 = counter4.long().unsqueeze(1)
        indcurr = indp_.gather(1, indp_.size(1) - 1 - counter2)
        b2 = (sb_.gather(1, counter2) - s_.gather(1, counter2) * p_.gather(1, indcurr)).squeeze(1)
        c = b_ - b2 > 0

        lb = torch.where(c, counter4, lb)
        ub = torch.where(c, ub, counter4)

    lb = lb.long()

    if c_l.any():
        lmbd_opt = torch.clamp_min((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), min=0).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = torch.clamp_min((b[c2] - sb[c2, lb]) / (-s[c2, lb]), min=0).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * a[c2] + torch.max(-lmbd_opt, d[c2]) * (1 - a[c2])

    return d * (w != 0).float()


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    # Probably the one of interest
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane
    # Change this?
    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1 # +1 and -1s
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    r = torch.max(t / w, (t - 1) / w).clamp(min=-1e12, max=1e12)
    r.masked_fill_(w.abs() < 1e-8, 1e12)
    r[r == -1e12] *= -1
    rs, indr = torch.sort(r, dim=1)
    rs2 = F.pad(rs[:, 1:], (0, 1))
    rs.masked_fill_(rs == 1e12, 0)
    rs2.masked_fill_(rs2 == 1e12, 0)
    # Change this?
    w3s = (w ** 2).gather(1, indr)
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w)
    d.mul_((w.abs() > 1e-8).float())
    # Change this?
    s = torch.cat((-w5 * rs[:, 0:1], torch.cumsum((-rs2 + rs) * ws, dim=1) - w5 * rs[:, 0:1]), 1)

    c4 = s[:, 0] + c < 0
    # Change this?
    c3 = (d * w).sum(dim=1) + c > 0
    c2 = ~(c4 | c3)

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) + c_ > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb = lb.long()

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        alpha[ws[c2, lb] == 0] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()


def null_space_pytorch(At, rcond=None): # Our modification
    # Taken from
    # https://discuss.pytorch.org/t/nullspace-of-a-tensor/69980/4
    # and then modified slightly
    # ut, st, vht = torch.Tensor.svd(At, some=False, compute_uv=True)
    # Add batch dim if doesn't exist
    if len(At.shape) != 3:
        At = At.unsqueeze(0)
    
    ut, st, vht = torch.svd(At, some=False, compute_uv=True)
    vht = vht.permute(0, 2, 1) # vht.T        
    Mt, Nt = ut.shape[1], vht.shape[2] # ut.shape[0], vht.shape[1]
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    
    tolt = (torch.max(st, dim=1)[0]*rcondt).unsqueeze(1) # torch.max(st)*rcondt
    numt = torch.sum(st > tolt, dtype=int, dim=1)
    # nullspace = vht[numt:,:].T.cpu().conj()
    nullspace = [vht[idx, x:, :].T.conj() for idx, x in enumerate(numt)]

    return nullspace


def projection_lsigma2(points_to_project, w_hyperplane, b_hyperplane, 
        ellipse_mat):
    '''
    We must project points to a plane, defined by `w` and `b`: w.T @ x + b, 
        under the norm induced by a matrix ('sigma' here). To do this, we
    (1): Express this plane, not as `w` and `b`, but as a minimal set of vectors
        that (upon translation) span this plane. This corresponds to a 
        'translated' version of the null space of `w`.
    (2): Solving a simple optimization problem of minimizing the distance to the
        plane w.r.t. the norm induced by the matrix. By equating the derivative
        of the norm to zero, we arrive at yet another minimization problem that
        we can solve through least squares.
    
    This procedure DOES NOT consider box-constraints (referring to the usual 
    pixel requirements of [0,1]^d)
    '''
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane
    # t.shape == w.shape == [2*bs, lat_dim], b.shape == [2*bs]
    lat_dim = t.shape[1]
    # Compute null space, which corresponds to the vectors which span each 
    # (origin-shifted) plane
    V = null_space_pytorch(w.unsqueeze(1))
    import pdb; pdb.set_trace()
    assert all([x.shape[0] == lat_dim and x.shape[1] == (lat_dim-1) 
        for x in V])
    V = torch.cat([x.unsqueeze(0) for x in V])
    # Find translation vector of the plane by projecting origin onto hyperplane
    coeff = - b / torch.norm(w, dim=1)**2
    tra = coeff.unsqueeze(1) * w # transl.shape == [2*bs, lat_dim]
    # Ensure translation vector is on hyperplane
    dist_to_hyp = torch.matmul(w.unsqueeze(1), tra.unsqueeze(2)).squeeze() + b
    assert torch.allclose(dist_to_hyp, torch.zeros_like(b), atol=1e-5)

    # With these variables, we can think it terms of vector `x`, full of coeffs
    # for the columns of V
    # Return from homogeneous coordinates (divide by last entry of each vector)
    coeffs = all_null_spaces[:, -1, :].unsqueeze(1)
    mod_null_space = all_null_spaces[:, :-1, :] / coeffs
    one_w = w[0]
    one_b = b[0]
    one_null_space = mod_null_space[0]
    # Compute null spaces of each `w`
    all_null_spaces = []
    for curr_w, curr_b in zip(w, b):
        # Compute current null space
        curr_nll = null_space_pytorch(curr_w.unsqueeze(1).T)
        all_null_spaces.append(curr_nll)

    all_null_spaces = torch.cat([x.unsqueeze(0) for x in all_null_spaces])
    # Now we have a system of the form
    # Sigma @ (V @ x - p) = 0
    # Where 
    # --> Sigma (ellipse_mat) is the matrix parameterizing the ellipse
    # --> V (all_null_spaces) is a matrix whose column vectors span the plane
    # --> x is the coefficients of the vectors in V
    # --> p (t) is the vector we want to project to the plane
    # To solve this, we use least squares
    targets = torch.matmul(ellipse_mat.unsqueeze(0), t.unsqueeze(2))
    # targets.shape == [2*bs, lat_dim, 1]
    A = torch.matmul(ellipse_mat.unsqueeze(0), all_null_spaces)
    # A.shape == [2*bs, lat_dim, lat_dim-1]
    # With these variables we have to solve A@x = targets through least squares
    x = torch.linalg.lstsq(A, targets).solution
    # x.shape == [2*bs, lat_dim-1, 1], which are the coefficients for each of
    # the vectors that span the plane. Thus, the projection is given by V @ x
    projs = torch.matmul(all_null_spaces, x)
    # projs.shape == [2*bs, lat_dim, 1]
    
    return projs


def projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    r = (1 / w).abs().clamp_max(1e12)
    indr = torch.argsort(r, dim=1)
    indr_rev = torch.argsort(indr)

    c6 = (w < 0).float()
    d = (-t + c6) * (w != 0).float()
    ds = torch.min(-w * t, w * (1 - t)).gather(1, indr)
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c2 = s[:, -1] < 0

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, s.shape[1])
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb2 = lb.long()

    if c2.any():
        indr = indr[c2].gather(1, lb2.unsqueeze(1)).squeeze(1)
        u = torch.arange(0, w.shape[0], device=device).unsqueeze(1)
        u2 = torch.arange(0, w.shape[1], device=device, dtype=torch.float).unsqueeze(0)
        alpha = -s[c2, lb2] / w[c2, indr]
        c5 = u2 < lb.unsqueeze(-1)
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * u3.float()
        d[c2, indr] = alpha

    return d * (w.abs() > 1e-8).float()
