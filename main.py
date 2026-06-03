import argparse
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma
from collections import namedtuple

parser = argparse.ArgumentParser("Simplified Kurihara VDP for DP-MoG")
parser.add_argument("--data_csv", type=str)
parser.add_argument("--cluster_observation_stddev", type=float, default=0.05)
parser.add_argument("--cluster_location_prior_stddev", type=float, default=1.0)
parser.add_argument("--cluster_location_posterior_stddev", type=float, default=0.05)
parser.add_argument("--inferred_clusters_limit", type=int, default=10)  # T, everything after is tied to priors
args = parser.parse_args()

BetaDistribution = namedtuple("BetaDistribution", ["alpha", "beta"])
GaussianDistribution = namedtuple("GaussianDistribution", ["mean", "stddev"])
InfCategoricalDistribution = namedtuple("InfCategoricalDistribution", ["headprobs", "tailsum"])
SniInfo = namedtuple("SniInfo", ["terms", "headsum", "tailsum"])

def get_pv():
    return BetaDistribution(alpha=1.0, beta=1.0)

def get_qv_initial():
    # sample a hyperprior on mean: alpha / (alpha + beta)
    mean = np.random.beta(1.1, 1.1, size=[args.inferred_clusters_limit]) # [T]
    # sample a hyperprior on concentration: (alpha + beta)
    conc = np.random.pareto(1.5, size=[args.inferred_clusters_limit])    # [T]
    # convert to (alpha, beta) variational params
    phi_v1 = mean * conc
    phi_v2 = conc - phi_v1
    return BetaDistribution(alpha=phi_v1, beta=phi_v2)

def get_peta():
    return GaussianDistribution(mean=0.0, stddev=args.cluster_location_prior_stddev)

def get_qeta_initial(data_dim):
    mu = np.random.normal(
        loc=0.0, 
        scale=args.cluster_location_prior_stddev, 
        size=[args.inferred_clusters_limit, data_dim],
    )
    return GaussianDistribution(mean=mu, stddev=args.cluster_location_posterior_stddev)

def get_px_given_eta(eta):
    return GaussianDistribution(mean=eta, stddev=args.cluster_observation_stddev)

def get_sni_info(
    xs, # [N, D]
    qv, # ([T], [T])
    qeta, # ([T, D], [])
    pv, # ([], [])
    peta, # ([], [])
):
    sigma_x = args.cluster_observation_stddev
    line_11 = digamma(qv.alpha) - digamma(qv.alpha + qv.beta)  # [T]
    line_12 = digamma(qv.beta) - digamma(qv.alpha + qv.beta)  # [T]
    line_13 = np.einsum('ld,nd->nl', qeta.mean / (sigma_x ** 2), xs) + \
        -0.5 * np.einsum('ld,ld->l', qeta.mean / (sigma_x ** 2), qeta.mean)[None, ...] # [N, T]
    
    S_n_i = (
        line_11[None,...] + 
        np.cumsum(np.concatenate([np.array([0]), line_12[:-1]], axis=0), axis=0)[None, ...] + 
        line_13  # [N, T]
    )
    exp_S_n_i = np.exp(S_n_i)  # [N, T]
    exp_S_n_headsum = np.sum(exp_S_n_i, axis=-1)  # [N]

    line_11_tp1 = digamma(pv.alpha) - digamma(pv.alpha + pv.beta)  # []
    line_12_tp1 = digamma(pv.beta) - digamma(pv.alpha + pv.beta)  # []
    line_13_tp1 = (
        np.einsum('d,nd->n', np.full(fill_value=peta.mean, shape=xs.shape[1]), xs)
        - xs.shape[1] * (peta.mean ** 2 + peta.stddev ** 2) / (sigma_x ** 2) # [N]
    )
    S_n_tp1 = line_11_tp1 + np.sum(line_12) + line_13_tp1  # [N]
    exp_S_n_tailsum = S_n_tp1 / (1 - np.exp(line_12_tp1))  # [N]
    return SniInfo(terms=exp_S_n_i, headsum=exp_S_n_headsum, tailsum=exp_S_n_tailsum)

def update_qz(sni_info):
    exp_S_n_i = sni_info.terms
    exp_S_n_sum = sni_info.headsum + sni_info.tailsum
    q_zi_head = exp_S_n_i / exp_S_n_sum[..., None]  # [N, T]
    q_zi_tailsum = 1 - np.sum(q_zi_head, axis=-1)
    return InfCategoricalDistribution(headprobs=q_zi_head, tailsum=q_zi_tailsum)

def update_qv(
    qz, # ([N, T], [N])
    pv, # ([], [])
):
    # compute line 14 left
    qv_phi_1_new = pv.alpha + np.sum(qz.headprobs, axis=0) # [T]
    # now compute sum_j={i+1}^infty = sum_j={i+1}^T + sum_j={T+1}^infty
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
    qz_ip1_tailsum = unflip + qz.tailsum[..., None]  # [N, T]
    # compute line 14 right
    qv_phi_2_new = pv.beta + np.sum(qz_ip1_tailsum, axis=0)  # [T]
    return BetaDistribution(alpha=qv_phi_1_new, beta=qv_phi_2_new)

def update_qeta(xs, qz, peta):
    return GaussianDistribution(
        mean=peta.mean + np.einsum('nt,nd->td', qz.headprobs, xs),  # [T, D], 
        stddev=args.cluster_location_posterior_stddev,
    )

def get_total_beta_kl_diverence(
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
    qeta, # ([T, D], [])
    peta, # ([], [])
):
    qeta_mu = qeta.mean
    qeta_sigma = np.full_like(qeta_mu, fill_value=qeta.stddev)
    peta_mu = np.full_like(qeta_mu, fill_value=peta.mean)
    peta_sigma = np.full_like(qeta_mu, fill_value=peta.stddev)

    term1 = np.log(peta_sigma / qeta_sigma)
    term2 = (qeta_sigma * qeta_sigma) / (2.0 * peta_sigma * peta_sigma)
    term3 = ((qeta_mu - peta_mu) * (qeta_mu - peta_mu)) / (2.0 * peta_sigma * peta_sigma)
    term4 = np.full_like(qeta_mu, fill_value=-0.5)
    kls = np.sum(term1 + term2 + term3 + term4, axis=-1)  # [T]
    return np.sum(kls, axis=0) # []

def get_elbo(xs, qv, qeta, pv, peta):
    # computes the elbo assuming q(z) was optimized last
    total_kl_beta = get_total_beta_kl_diverence(qv, pv)
    total_kl_gauss = get_total_gaussian_kl_divergence(qeta, peta)
    sni_info = get_sni_info(xs, qv, qeta, pv)
    sn_infsum = sni_info.headsum + sni_info.tailsum
    free_energy = total_kl_beta + total_kl_gauss - np.sum(np.log(sn_infsum), axis=0)
    elbo = -free_energy
    return elbo

def main():
    args = parser.parse_args()
    # df = pd.read_csv(args.data_csv)
    # xs = df.to_numpy()

    xs = np.random.normal(loc=0.5, scale=0.1, shape=[100, 1])
    pv = get_pv()
    peta = get_peta()
    qv = get_qv_initial()
    qeta = get_qeta_initial(data_dim=xs.shape[1])

    qz = update_qz(get_sni_info(xs, qv, qeta, pv, peta))
    print(f"ELBO: {get_elbo(xs, qv, qeta, pv, peta)}")

    for _ in range(0, 10):
        qv = update_qv(qz, pv)
        qeta = update_qeta(xs, qz, peta)
        qz = update_qz(get_sni_info(xs, qv, qeta, pv, peta))
        print(f"ELBO: {get_elbo(xs, qv, qeta, pv, peta)}")
