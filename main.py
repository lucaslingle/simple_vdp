import argparse
import numpy as np
import pandas as pd
from scipy.special import digamma

parser = argparse.ArgumentParser("Simplified Kurihara VDP for DP-MoG")
parser.add_argument("--data_csv", type=str)
parser.add_argument("--cluster_observation_stddev", type=float, default=0.05)
parser.add_argument("--cluster_location_prior_stddev", type=float, default=1.0)
parser.add_argument("--cluster_location_posterior_stddev", type=float, default=0.05)
parser.add_argument("--inferred_clusters_limit", type=int, default=10)  # T, everything after is tied to priors
args = parser.parse_args()

def get_pv_params():
    return 1.0, 1.0

def get_qv_params_init():
    # sample a hyperprior on mean: alpha / (alpha + beta)
    mean = np.random.beta(1.1, 1.1, size=[args.inferred_clusters_limit]) # [T]
    # sample a hyperprior on concentration: (alpha + beta)
    conc = np.random.pareto(1.5, size=[args.inferred_clusters_limit])    # [T]
    # convert to (alpha, beta) variational params
    phi_v1 = mean * conc
    phi_v2 = conc - phi_v1
    return phi_v1, phi_v2

def get_qeta_phi1_init(data_dim):
    phi_eta1 = np.random.normal(
        loc=0.0, 
        scale=args.cluster_location_prior_stddev, 
        size=[args.inferred_clusters_limit, data_dim],
    )
    return phi_eta1

def get_sni_info(
    xs, # [N, D]
    qv_phi_1, # [T]
    qv_phi_2, # [T]
    qeta_phi_1, # [T, D]
    pv_alpha_1, # []
    pv_alpha_2, # []
):
    sigma_x = args.cluster_observation_stddev
    line_11 = digamma(qv_phi_1) - digamma(qv_phi_1 + qv_phi_2)  # [T]
    line_12 = digamma(qv_phi_2) - digamma(qv_phi_1 + qv_phi_2)  # [T]
    line_13 = np.einsum('ld,nd->nl', qeta_phi_1 / (sigma_x ** 2), xs) + \
        -0.5 * np.einsum('ld,ld->l', qeta_phi_1 / (sigma_x ** 2), qeta_phi_1)[None, ...] # [N, T]
    
    S_n_i = line_11[..., None] + \
        np.cumsum(np.concatenate([np.array([0]), line_12[:-1]], axis=0), axis=0)[None, ...] + \
        line_13  # [N, T]
    exp_S_n_i = np.exp(S_n_i)  # [N, T]
    exp_S_n_headsum = np.sum(exp_S_n_i, axis=-1)  # [N]

    line_11_tp1 = digamma(pv_alpha_1) - digamma(pv_alpha_1 + pv_alpha_2)  # []
    line_12_tp1 = digamma(pv_alpha_2) - digamma(pv_alpha_1 + pv_alpha_2)  # []
    line_13_tp1 = 0.0  # zero because qeta_phi_1 for T+1 is set to prior's mean of zero
    S_n_tp1 = line_11_tp1 + np.sum(line_12) + line_13_tp1  # []
    exp_S_n_tailsum = S_n_tp1 / (1 - np.exp(line_12_tp1))  # []

    return exp_S_n_i, exp_S_n_headsum, exp_S_n_tailsum
    
def update_qz(exp_S_n_i, exp_S_n_sum):
    q_zi_head = exp_S_n_i / exp_S_n_sum[..., None]  # [N, T]
    return q_zi_head

def update_qv(
    q_zi_head, # [N, T]
    q_zi_tailsum, # [N]
    pv_alpha_1, # []
    pv_alpha_2, #[]
):
    qv_phi_1_new = pv_alpha_1 + np.sum(q_zi_head, axis=0) # [T]

    # compute sum_j={i+1}^infty = sum_j={i+1}^T + sum_j={T+1}^infty
    # i=1 -> sum i=2 ... i=T
    # ...
    # i=T-2 -> sum i=T-1 ... i=T
    # i=T-1 -> sum i=T
    # i=T -> 0
    N = q_zi_head.shape[0]
    chop = q_zi_head[:, 1:]
    flip = chop[:, ::-1]
    pad = np.pad(flip, ((0, 0), (1, 0)), mode='constant')
    cumulative = np.cumsum(pad, axis=-1)
    unflip = cumulative[:, ::-1]  # [N, T]
    qz_ip1_tailsum = unflip + q_zi_tailsum  # [N, T]

    qv_phi_2_new = pv_alpha_2 + np.sum(qz_ip1_tailsum, axis=0)  # [T]
    
    return qv_phi_1_new, qv_phi_2_new


def main():
    args = parser.parse_args()
    df = pd.read_csv(args.data_csv)
    xs = df.to_numpy()
