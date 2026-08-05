from collections import namedtuple
import numpy as np
from scipy.special import gammaln, digamma
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR, force=True)
logger = logging.getLogger(__name__)

BetaDist = namedtuple("BetaDist", ["alpha", "beta"])
GaussianDist = namedtuple("GaussianDist", ["mean", "stddev"])
CategoricalDist = namedtuple("CategoricalDist", ["headprobs", "headsum", "stabilizer"])

TRUNCATION_LEVEL = 10
SIGMA_C = 1.0
SIGMA_X = 0.05
NUM_DATA = 2000
DIM_DATA = 2

def get_pc():
    return GaussianDist(mean=0.0, stddev=SIGMA_C)

def get_qc_initial(data_dim):
    mu = np.random.normal(
        loc=0.0, 
        scale=SIGMA_C, 
        size=[TRUNCATION_LEVEL, data_dim],
    )
    sigma = np.repeat(np.array([SIGMA_C]), repeats=TRUNCATION_LEVEL, axis=0)
    return GaussianDist(mean=mu, stddev=sigma)

def get_pv():
    return BetaDist(alpha=1.0, beta=1.0)

def get_qv_initial():
    # sample a hyperprior on mean: alpha / (alpha + beta)
    mean = np.random.beta(1.1, 1.1, size=[TRUNCATION_LEVEL]) # [T]
    # sample a hyperprior on concentration: (alpha + beta)
    conc = np.random.pareto(1.5, size=[TRUNCATION_LEVEL])    # [T]
    # convert to (alpha, beta) variational params
    alpha = mean * conc
    beta = conc - alpha
    return BetaDist(alpha=alpha, beta=beta)

def update_qz(
    *,
    xs, # [N, D]
    qc, # ([T, D], [])
    qv, # ([T], [T])
):
    line_11 = digamma(qv.alpha) - digamma(qv.alpha + qv.beta)  # [T]
    line_12 = digamma(qv.beta) - digamma(qv.alpha + qv.beta)  # [T]
    line_13 = np.einsum('ld,nd->nl', qc.mean / (SIGMA_X ** 2), xs) + \
        -0.5 * np.einsum('ld,ld->l', qc.mean / (SIGMA_X ** 2), qc.mean)[None, ...] # [N, T]
    S_n_i = (
        line_11[None,...] + 
        np.cumsum(np.concatenate([np.array([0]), line_12[:-1]], axis=0), axis=0)[None, ...] + 
        line_13
    ) # [N, T]
    stabilizer = np.max(S_n_i, axis=-1)                # [N]
    exp_S_n_i = np.exp(S_n_i - stabilizer[..., None])  # [N, T]
    exp_S_n_headsum = np.sum(exp_S_n_i, axis=-1)       # [N]
    q_zn_head = exp_S_n_i / exp_S_n_headsum[..., None]  # [N, T]
    return CategoricalDist(headprobs=q_zn_head, headsum=exp_S_n_headsum, stabilizer=stabilizer)

def update_qv(
    *,
    qz, # ([N, T], [N])
    pv, # ([], [])
):
    qv_nu_1 = (pv.alpha - 1) + np.sum(qz.headprobs, axis=0) # [T]
    # now compute sum_j={i+1}^T
    # i=1 -> sum i=2 ... i=T
    # ...
    # i=T-2 -> sum i=T-1 ... i=T
    # i=T-1 -> sum i=T
    # i=T -> 0
    N = qz.headprobs.shape[0]
    chop = qz.headprobs[:, 1:]
    flip = chop[:, ::-1]
    pad = np.pad(flip, ((0, 0), (1, 0)), mode='constant')
    cumulative = np.cumsum(pad, axis=-1)
    unflip = cumulative[:, ::-1]  # [N, T]
    qv_nu_2 = (pv.beta - 1) + np.sum(unflip, axis=0)  # [T]
    return BetaDist(alpha=qv_nu_1 + 1, beta=qv_nu_2 + 1)

def update_qc(*, xs, qz):
    numer = (SIGMA_X ** -2) * np.einsum('nt,nd->td', qz.headprobs, xs)
    denom = (SIGMA_C ** -2) + (SIGMA_X ** -2) * np.sum(qz.headprobs, axis=0)
    return GaussianDist(
        mean=numer / denom[..., None],
        stddev=denom ** -0.5,
    )

def get_total_beta_kl_diverence(
    *,
    qv, # ([T], [T])
    pv, # ([], [])
):
    log_beta_q = gammaln(qv.alpha + qv.beta) - gammaln(qv.alpha) - gammaln(qv.beta)
    log_beta_p = gammaln(pv.alpha + pv.beta) - gammaln(pv.alpha) - gammaln(pv.beta)
    term_normalization = log_beta_q - log_beta_p

    term_expectation = (
        (qv.alpha - pv.alpha) * (digamma(qv.alpha) - digamma(qv.alpha + qv.beta)) +
        (qv.beta - pv.beta) * (digamma(qv.beta) - digamma(qv.alpha + qv.beta))
    )

    return np.sum(term_normalization + term_expectation, axis=0)

def get_total_gaussian_kl_divergence(
    *,
    qc, # ([T, D], [T])
    pc, # ([], [])
):
    qc_mu = qc.mean
    qc_sigma = np.repeat(qc.stddev[..., None], repeats=qc.mean.shape[-1], axis=1)
    pc_mu = np.full_like(qc_mu, fill_value=pc.mean)
    pc_sigma = np.full_like(qc_mu, fill_value=pc.stddev)

    term1 = np.log(pc_sigma / qc_sigma)
    term2 = (qc_sigma * qc_sigma) / (2.0 * pc_sigma * pc_sigma)
    term3 = ((qc_mu - pc_mu) * (qc_mu - pc_mu)) / (2.0 * pc_sigma * pc_sigma)
    term4 = np.full_like(qc_mu, fill_value=-0.5)
    kls = np.sum(term1 + term2 + term3 + term4, axis=-1)  # [T]
    return np.sum(kls, axis=0) # []

def get_elbo_normalized(*, xs, qc, qv, pc, pv):
    # computes the elbo/(N*D), and assumes q(z) was optimized last
    N = xs.shape[0]
    D = xs.shape[1]

    kl_gauss = get_total_gaussian_kl_divergence(qc=qc, pc=pc) / (N * D)
    logger.info(f"kl_gauss: {kl_gauss}")

    kl_beta = get_total_beta_kl_diverence(qv=qv, pv=pv) / (N * D)
    logger.info(f"kl_beta: {kl_beta}")

    qz = update_qz(xs=xs, qc=qc, qv=qv)
    assert qz.headsum.shape == (N,)
    assert qz.stabilizer.shape == (N,)
    sn_infsum = qz.headsum
    stabilizer = qz.stabilizer
    lastterm = -np.mean(stabilizer + np.log(sn_infsum), axis=0) / (D)
    logger.info(f"lastterm: {lastterm}")

    free_energy = kl_beta + kl_gauss + lastterm
    logger.info(f"free_energy: {free_energy}")

    elbo = -free_energy
    logger.info(f"elbo: {elbo}")
    return elbo

def get_mean_stick_lengths(*, qv):
    means = qv.alpha / (qv.alpha + qv.beta)  # [T]
    minus = 1 - means  # [T]
    minus_prod = np.cumprod(minus, axis=0)  # [T]
    minus_prod = np.pad(minus_prod[0:-1], ((1, 0)), mode='constant', constant_values=1.0)
    return means * minus_prod

def permute_cluster_ids(*, qc, qv, qz):
    stick_means = get_mean_stick_lengths(qv=qv)
    sort_idxs = np.argsort(stick_means)[::-1]
    qc_new = GaussianDist(
        mean=np.take_along_axis(qc.mean, sort_idxs[..., None], axis=0), 
        stddev=np.take_along_axis(qc.stddev, sort_idxs, axis=0),
    )
    qv_new = BetaDist(
        alpha=np.take_along_axis(qv.alpha, sort_idxs, axis=0), 
        beta=np.take_along_axis(qv.beta, sort_idxs, axis=0), 
    )
    qz_new = CategoricalDist(
        headprobs=np.take_along_axis(qz.headprobs, sort_idxs[None, ...], axis=1), 
        headsum=np.take_along_axis(qz.headsum, sort_idxs, axis=0),
        stabilizer=qz.stabilizer,
    )
    return qc_new, qv_new, qz_new

def print_stick_lengths(*, qv):
    stick_means = get_mean_stick_lengths(qv=qv)
    for i in range(qv.alpha.shape[0]):
        print(stick_means[i])

def main():
    np.random.seed(42)
    xs0 = np.random.normal(loc=0.5, scale=0.05, size=[NUM_DATA // 2, DIM_DATA])
    xs1 = np.random.normal(loc=-0.5, scale=0.05, size=[NUM_DATA // 2, DIM_DATA])
    xs = np.concatenate([xs0, xs1], axis=0)

    pv = get_pv()
    pc = get_pc()
    qv = get_qv_initial()
    qc = get_qc_initial(data_dim=xs.shape[1])

    qz = update_qz(xs=xs, qc=qc, qv=qv)
    elbo = get_elbo_normalized(xs=xs, qc=qc, qv=qv, pc=pc, pv=pv)
    print(f"ELBO normalized: {elbo}")

    for _ in range(0, 10):
        qc, qv, qz = permute_cluster_ids(qc=qc, qv=qv, qz=qz)
        qc = update_qc(xs=xs, qz=qz)
        qv = update_qv(qz=qz, pv=pv)
        qz = update_qz(xs=xs, qc=qc, qv=qv)
        elbo = get_elbo_normalized(xs=xs, qc=qc, qv=qv, pc=pc, pv=pv)
        print(f"ELBO normalized: {elbo}")

    print_stick_lengths(qv=qv)

    # ppd = get_posterior_predictive_density(qv, qeta, peta)
    # plot_ppd2d(ppd)

if __name__ == "__main__":
    main()