from math import ceil

import numpy as np
import torch
from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint

from .certificate import Certificate


class Smooth():
    """A smoothed classifier g
    Adapted from:
            https://github.com/locuslab/smoothing/blob/master/code/core.py
        to use an arbitrary certificate Certificate
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(
            self, base_classifier: torch.nn.Module, num_classes: int,
            sigma: torch.Tensor, certificate: Certificate
    ):
        """
        Args:
            base_classifier (torch.nn.Module): maps from
                [batch x latent_dimension]
            to [batch x num_classes]
            num_classes (int): number of classes
            sigma (torch.Tensor): distribution parameter
            certificate (Certificate): certificate desired
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.certificate = certificate

    def certify(
            self, z: torch.tensor, x: torch.tensor, label: torch.tensor, 
            n0: int, n: int, alpha: float, batch_size: int, 
            device: torch.device=torch.device('cuda:0')
    ):
        """Monte Carlo algorithm for certifying that g's prediction around x
        is constant within L2 radius.
        With probability at least 1 - alpha, the class returned by this method
        will equal g(x), and g's prediction will robust within a L2 ball of
        radius R around z for some semantic directions perturbed by x.
        Args:
            z (torch.tensor): is the latent code for a person
            x (torch.tensor): the input (amount of pertuations to the semantic directions)
            n0 (int): the number of Monte Carlo samples to use for selection (usually it is 100)
            n (int): the number of Monte Carlo samples to use for estimation (usually is is 100,000)
            alpha (float): the failure probability (usually it is 0.001)
            batch_size (int): batch size to use when evaluating the base classifier
            device (torch.device, optional):
        Returns:
            int, float: (predicted class, gap term in the certified radius)
            in the case of abstention, the class will be ABSTAIN and the
            radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(z, x, n0, batch_size, device=device)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        if cAHat != label.item():
            return cAHat, 0.0
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(z, x, n, batch_size, device=device)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, self.certificate.compute_gap(pABar)

    def predict(
            self, z: torch.tensor, x: torch.tensor, n: int, alpha: float, batch_size: int,
            device: torch.device = torch.device('cuda:0')
    ) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at (z, x).
        With probability at least 1 - alpha, the
        class returned by this method will equal g(z, x).
        This function uses the hypothesis test described in
        https://arxiv.org/abs/1610.03944 for identifying the top category of
        a multinomial distribution.
        Args:
            z (torch.tensor): is the latent code for a person
            x (torch.tensor): the input (amount of pertuations to the semantic directions)
            n (int): the number of Monte Carlo samples to use
            alpha (float): the failure probability
            batch_size (int): batch size to use when evaluating the base classifier
            device (torch.device, optional): device on which to perform the computations
        Returns:
            int: output class
        """
        self.base_classifier.eval()
        counts = self._sample_noise(z, x, n, batch_size, device=device)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(
            self, z: torch.tensor, x: torch.tensor, num: int, batch_size,
            device: torch.device = torch.device('cuda:0')
    ) -> np.ndarray:
        """Sample the base classifier's prediction under noisy corruptions of
        the input x.
        Args:
            z (torch.tensor): is the latent code for a person
            x (torch.tensor): the input (amount of pertuations to the semantic directions)
            num (int): number of samples to collect
            batch_size (TYPE): Description
            device (torch.device, optional): device on which to perform the
                computations
        Returns:
            np.ndarray: an ndarray[int] of length num_classes containing the
            per-class counts
        """
        with torch.no_grad():
            # counts = np.zeros(self.num_classes, dtype=int)
            counts = torch.zeros(self.num_classes, dtype=float, device=device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = self.certificate.sample_noise(batch, self.sigma)
                predictions = self.base_classifier(z, batch + noise).argmax(1)
                counts += self._count_arr(predictions,
                                          device, self.num_classes)
            return counts.cpu().numpy()

    def _count_arr(
            self, arr: torch.tensor, device: torch.device, length: int
    ) -> torch.tensor:
        counts = torch.zeros(length, dtype=torch.long, device=device)
        unique, c = arr.unique(sorted=False, return_counts=True)
        counts[unique] = c
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli
        proportion.
        This function uses the Clopper-Pearson method.
        Args:
            NA (int): the number of "successes"
            N (int): the number of total draws
            alpha (float): the confidence level
        Returns:
            float: a lower bound on the binomial proportion which holds true
            w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]