# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import torch

from autoattack.fab_projections import projection_linf, projection_l2,\
    projection_l1, projection_lsigma2 # Our modification

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}

# Modified by us
from attack_utils.proj_utils import sq_distance
from attack_utils.gen_utils import (init_deltas, ELLIPSE_MAT, RED_ELLIPSE_MAT,
    ELLIPSE_MAT_INV, RED_ELLIPSE_MAT_INV)

class FABAttack():
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9,
            lin_comb=False):
        """ FAB-attack implementation in pytorch """

        self.norm = norm
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps if eps is not None else DEFAULT_EPS_DICT_BY_NORM[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.targeted = False
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        self.lin_comb = lin_comb
        self.mat = RED_ELLIPSE_MAT if self.lin_comb else ELLIPSE_MAT
        self.mat_inv = RED_ELLIPSE_MAT_INV if self.lin_comb else ELLIPSE_MAT_INV

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def _predict_fn(self, x):
        raise NotImplementedError("Virtual function.")

    def _get_predicted_label(self, x):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch(self, imgs, la):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
       raise NotImplementedError("Virtual function.")

    def attack_single_run(self, x, y=None, use_rand_start=False, 
            is_targeted=False):
        """
        :param x:             clean images
        :param y:             clean labels, if None we use the predicted labels
        :param is_targeted    True if we ise targeted version. Targeted class is 
                                assigned by `self.target_class`
        """
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        #assert next(self.predict.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        if is_targeted:
            output = self._predict_fn(x)
            la_target = output.sort(dim=-1)[1][:, -self.target_class]
            la_target2 = la_target[pred].detach().clone()

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            use_rand_start = True # Hard-code this here
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.reshape([t.shape[0], -1]).abs()
                                         .max(dim=1, keepdim=True)[0]
                                         .reshape([-1, *[1]*self.ndims])) * .5
                elif self.norm == 'L1':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.abs().view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .view(t.shape[0], *[1]*self.ndims)) / 2
                elif self.norm == 'L2': # Probably like the one of interest
                    '''
                    I think all this code can be replaced with our sampling
                    '''
                    # Sample some random noise
                    t = torch.randn(x1.shape).to(self.device)
                    # Normalize this noise in the L2 sense
                    coeff = (t ** 2).view(t.shape[0], -1).sum(dim=-1).sqrt()
                    coeff = coeff.view(t.shape[0], *[1]*self.ndims)
                    # The size of the step (since t is normalized)
                    term1 = self.eps * torch.ones(res2.shape).to(self.device)
                    minn = torch.min(res2, term1).reshape([-1, *[1]*self.ndims])
                    # Compute the step and add it
                    step = minn * t / coeff * .5
                    x1 = im2 + step
                elif self.norm == 'Lsigma2': # Our modification
                    # Sample some random noise (hardcoded on_surface)
                    deltas = init_deltas(
                        random_init=True, lin_comb=self.lin_comb, 
                        n_vecs=x.size(0), on_surface=True
                    )
                    # Add the step
                    x1 = im2 + deltas.unsqueeze(2).unsqueeze(3)
                
                # This clamping assumes we're dealing with images!
                # x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with torch.no_grad():
                    # For computing 's' in the FAB paper
                    if is_targeted:
                        df, dg = self.get_diff_logits_grads_batch_targeted(x1, 
                            la2, la_target2)
                    else:
                        # I don't think it's computationally feasible to run the
                        # untargeted version of the attack
                        df, dg = self.get_diff_logits_grads_batch(x1, la2)
                        # df.shape == [bs, n_classes]
                        # dg.shape == if lin_comb: [bs, n_classes, N_DIRS, 1, 1]
                        #   else [bs, n_classes, lat_dim, 1, 1]
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 +
                                            dg.abs()
                                            .reshape(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1))
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                    elif self.norm == 'L2': # Probably like the one of interest
                        # I think the computation on dg should take into account
                        # the ellipse matrix
                        coeff = (dg ** 2).reshape(dg.shape[0], dg.shape[1], -1)
                        coeff = coeff.sum(dim=-1).sqrt()
                        dist1 = df.abs() / (coeff + 1e-12)
                    elif self.norm == 'Lsigma2': # Our modification
                        # I think the computation on dg should take into account
                        # the ellipse matrix
                        # Remove dims for simulating ims
                        temp_dg = dg.squeeze(4).squeeze(3)
                        # Combine dimensions
                        bs, n_cls = dg.size(0), dg.size(1)
                        temp_dg = temp_dg.reshape(bs * n_cls, -1, 1)
                        # Compute the distances
                        coeff = torch.sqrt(sq_distance(self.mat, temp_dg))
                        # Segment dimensions
                        coeff = coeff.reshape(bs, n_cls)
                        dist1 = df.abs() / (coeff + 1e-12)
                    else:
                        raise ValueError('norm not supported')
                    
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    # I think this is a dot product, and so it should take into 
                    # account the ellipse matrix
                    # Remove dims for simulating images
                    # term_add = (dg2 * x1).reshape(x1.shape[0], -1).sum(dim=-1)
                    term_add = sq_distance(self.mat, dg2.squeeze(3), x1.squeeze(3))
                    b = -df[u1, ind] + term_add
                    w = dg2.reshape([bs, -1])

                    # Exploit batch dim to compute vectors, simultaneously, for 
                    # the current iterate and the original one. The first 'bs'
                    # entries will correspond to the current iterate, while the 
                    # last 'bs' will correspond to the original one
                    if self.norm == 'Linf':
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L1':
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L2': # Probably the one of interest
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'Lsigma2': # Probably the one of interest
                        d3 = projection_lsigma2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                            self.mat_inv)
                        # d3.shape == [2*bs, lat_dim]
                    
                    # This is when the code extracts the d's for the current and
                    # original iterates
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    # dgX.shape == if lin_comb: [bs, n_classes, N_DIRS, 1, 1]
                    #   else [bs, n_classes, lat_dim, 1, 1]

                    if self.norm == 'Linf':
                        a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(dim=1, keepdim=True)\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L2': # Probably the one of interest
                        # I think the operation on d3 should take into account 
                        # the ellipse matrix
                        a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()
                        a0 = a0.view(-1, *[1]*self.ndims)
                    elif self.norm == 'Lsigma2': # Probably the one of interest
                        # I think the operation on d3 should take into account 
                        # the ellipse matrix
                        # a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()
                        a0 = sq_distance(self.mat, d3.unsqueeze(2)).sqrt()
                        a0 = a0.view(-1, *[1]*self.ndims)
                    
                    # a0 = torch.max(a0, 1e-8 * torch.ones(a0.shape))
                    a0 = torch.max(a0, 1e-8 * 
                        torch.ones(a0.shape, device=self.device))
                    a0 = a0.to(self.device)
                    # 'a1' is delta^{(i)} in the paper
                    a1 = a0[:bs]
                    # 'a2' is delta^{(i)}_{orig} in the paper
                    a2 = a0[-bs:]
                    # Apply something like a ReLU
                    zeross = torch.zeros(a1.shape, device=self.device)
                    term1 = torch.max(a1 / (a1 + a2), zeross)
                    term1 = term1.to(self.device)
                    # This is Eqn. (9) in the paper
                    to_comp = self.alpha_max * torch.ones(a1.shape)
                    alpha = torch.min(term1, to_comp.to(self.device))

                    d1_step = (x1 + self.eta * d1)
                    d2_step = (im2 + self.eta * d2)
                    x1 = d1_step * (1 - alpha) + d2_step * alpha
                    # This clamping assumes we're dealing with images!
                    # x1 = x1.clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == 'Linf':
                            t = (x1[ind_adv] - im2[ind_adv]).reshape(
                                [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                        elif self.norm == 'L1':
                            t = (x1[ind_adv] - im2[ind_adv])\
                                .abs().reshape(ind_adv.shape[0], -1).sum(dim=-1)
                        elif self.norm == 'L2': # Probably the one of interest
                            # I think the operation on x1 should take into 
                            # account the ellipse matrix
                            diff = x1[ind_adv] - im2[ind_adv]
                            t = (diff ** 2).reshape(ind_adv.shape[0], -1)
                            t = t.sum(dim=-1).sqrt()
                        elif self.norm == 'Lsigma2': # Our modification
                            diff = x1[ind_adv] - im2[ind_adv]
                            # t = (diff ** 2).reshape(ind_adv.shape[0], -1)
                            # t = t.sum(dim=-1).sqrt()
                            t = sq_distance(self.mat, diff.squeeze(3)).sqrt()
                        
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        if self.device is None:
            self.device = x.device
        adv = x.clone()
        with torch.no_grad():
            acc = self._predict_fn(x).max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            if not self.targeted:
                # Initialize best perturbation as inf
                best_res = float('inf') * torch.ones_like(y)
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: 
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x.clone() # x[ind_to_fool].clone()
                        y_to_fool = y.clone() # y[ind_to_fool].clone()
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, 
                            use_rand_start=(counter > 0), is_targeted=False)

                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs()
                            res = res.reshape(x_to_fool.shape[0], -1).max(1)[0]
                        elif self.norm == 'L1':
                            res = (x_to_fool - adv_curr).abs()
                            res = res.reshape(x_to_fool.shape[0], -1).sum(-1)
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2)
                            res = res.reshape(x_to_fool.shape[0], -1)
                            res = res.sum(dim=-1).sqrt()
                        elif self.norm == 'Lsigma2':
                            # res = ((x_to_fool - adv_curr) ** 2)
                            res = x_to_fool - adv_curr
                            # res = res.reshape(x_to_fool.shape[0], -1)
                            # res = res.sum(dim=-1).sqrt()
                            res = sq_distance(self.mat, res.squeeze(3)).sqrt()
                        
                        temp_pred = self._predict_fn(adv_curr).max(1)[1]
                        acc_curr = temp_pred == y_to_fool
                        where_fool = ~acc_curr # Where the system is fooled
                        where_better = res < best_res # Where pert is smaller
                        # Update adversary where both things hapen
                        where_both = where_fool & where_better
                        adv[where_both] = adv_curr[where_both].clone()
                        # Update best residual
                        best_res[where_both] = res[where_both]
                        # This computes acc as if there's a bound on input!
                        # acc_curr = torch.max(acc_curr, res > self.eps)

                        # ind_curr = (acc_curr == 0).nonzero().squeeze()
                        # acc[ind_to_fool[ind_curr]] = 0
                        acc[where_both] = acc_curr[where_both]
                        # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), self.eps, time.time() - startt))

            else:
                # Initialize best perturbation as inf
                best_res = float('inf') * torch.ones_like(y)
                for target_class in range(2, self.n_target_classes + 2):
                    self.target_class = target_class
                    for counter in range(self.n_restarts):
                        ind_to_fool = acc.nonzero().squeeze()
                        if len(ind_to_fool.shape) == 0: 
                            ind_to_fool = ind_to_fool.unsqueeze(0)
                        if ind_to_fool.numel() != 0:
                            x_to_fool = x.clone() # x[ind_to_fool].clone()
                            y_to_fool = y.clone() # y[ind_to_fool].clone()
                            adv_curr = self.attack_single_run(
                                x_to_fool, y_to_fool, 
                                use_rand_start=(counter > 0), is_targeted=True
                            )
                            if self.norm == 'Linf':
                                res = (x_to_fool - adv_curr).abs()
                                res = res.reshape(x_to_fool.shape[0], -1)
                                res = res.max(1)[0]
                            elif self.norm == 'L1':
                                res = (x_to_fool - adv_curr).abs()
                                res = res.reshape(x_to_fool.shape[0], -1)
                                res = res.sum(-1)
                            elif self.norm == 'L2':
                                res = ((x_to_fool - adv_curr) ** 2)
                                res = res.reshape(x_to_fool.shape[0], -1)
                                res = res.sum(dim=-1).sqrt()
                            elif self.norm == 'Lsigma2':
                                # res = ((x_to_fool - adv_curr) ** 2)
                                res = x_to_fool - adv_curr
                                # res = res.reshape(x_to_fool.shape[0], -1)
                                # res = res.sum(dim=-1).sqrt()
                                res = sq_distance(self.mat, res.squeeze(3))
                                res = res.sqrt()

                            temp_pred = self._predict_fn(adv_curr).max(1)[1]
                            acc_curr = temp_pred == y_to_fool
                            where_fool = ~acc_curr # Where the system is fooled
                            where_better = res < best_res # Where pert is smaller
                            # Update adversary where both things hapen
                            where_both = where_fool & where_better
                            adv[where_both] = adv_curr[where_both].clone()
                            # Update best residual
                            best_res[where_both] = res[where_both]
                            # This computes acc as if there's a bound on input!
                            # acc_curr = torch.max(acc_curr, res > self.eps)

                            # ind_curr = (acc_curr == 0).nonzero().squeeze()
                            # acc[ind_to_fool[ind_curr]] = 0
                            acc[where_both] = acc_curr[where_both]
                            # adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                            if self.verbose:
                                print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                    counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))

        return adv
