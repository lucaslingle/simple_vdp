from collections import namedtuple
import numpy as np
from scipy.special import gammaln, digamma
import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.ERROR, force=True)

BetaDist = namedtuple("BetaDist", ["alpha", "beta"])
GaussianDist = namedtuple("GaussianDist", ["mean", "stddev"])
CategoricalDist = namedtuple("CategoricalDist", ["headprobs", "headsum", "stabilizer"])


TRUNCATION_LEVEL = 10
SIGMA_C = 1.0
SIGMA_X = 0.05
NUM_DATA = 2000
DIM_DATA = 2


class SimpleVI:
    def __init__(
        self, 
        truncation_level: int, 
        sigma_c: float,
        sigma_x: float,
        xs: np.ndarray,
        kappa: float,
    ):
        self._truncation_level = truncation_level
        self._sigma_c = sigma_c
        self._sigma_x = sigma_x
        self._xs = xs
        self._kappa = kappa
        self._logger = logging.getLogger(__name__)

    def get_pc(self):
        return GaussianDist(mean=0.0, stddev=self._sigma_c)

    def get_qc_initial(self):
        mu = np.random.normal(
            loc=0.0, 
            scale=self._sigma_c, 
            size=[self._truncation_level, self._xs.shape[-1]],
        )
        sigma = np.repeat(np.array([self._sigma_c]), repeats=self._truncation_level, axis=0)
        return GaussianDist(mean=mu, stddev=sigma)

    def get_pv(self):
        return BetaDist(alpha=1.0, beta=1.0)

    def get_qv_initial(self):
        # sample a hyperprior on mean: alpha / (alpha + beta)
        mean = np.random.beta(1.1, 1.1, size=[self._truncation_level]) # [T]
        # sample a hyperprior on concentration: (alpha + beta)
        conc = np.random.pareto(1.5, size=[self._truncation_level])    # [T]
        # convert to (alpha, beta) variational params
        alpha = mean * conc
        beta = conc - alpha
        return BetaDist(alpha=alpha, beta=beta)

    def update_qz(
        self,
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
        self,
        *,
        qv,
        qz, # ([N, T], [N])
        pv, # ([], [])
        kappa,
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
        
        qv_nu_1 = kappa * qv_nu_1 + (1 - kappa) * (qv.alpha - 1)
        qv_nu_2 = kappa * qv_nu_2 + (1 - kappa) * (qv.beta - 1)
        return BetaDist(alpha=qv_nu_1 + 1, beta=qv_nu_2 + 1)

    def update_qc(self, *, qc, xs, qz, kappa):
        qc_gamma_1 = (self._sigma_x ** -2) * np.einsum('nt,nd->td', qz.headprobs, xs)
        qc_gamma_2 = (self._sigma_c ** -2) + (self._sigma_x ** -2) * np.sum(qz.headprobs, axis=0)
        
        qc_gamma_1 = kappa * qc_gamma_1 + (1 - kappa) * (qc.mean * qc.stddev[..., None] ** -2)
        qc_gamma_2 = kappa * qc_gamma_2 + (1 - kappa) * (qc.stddev ** -2)

        return GaussianDist(
            mean=qc_gamma_1 / qc_gamma_2[..., None],
            stddev=qc_gamma_2 ** -0.5,
        )

    def get_total_beta_kl_diverence(
        self,
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
        self,
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

    def get_elbo_normalized(self, *, xs, qc, qv, pc, pv):
        # computes the elbo/(N*D), and assumes q(z) was optimized last
        N = xs.shape[0]
        D = xs.shape[1]

        kl_gauss = self.get_total_gaussian_kl_divergence(qc=qc, pc=pc) / (N * D)
        self._logger.info(f"kl_gauss: {kl_gauss}")

        kl_beta = self.get_total_beta_kl_diverence(qv=qv, pv=pv) / (N * D)
        self._logger.info(f"kl_beta: {kl_beta}")

        qz = self.update_qz(xs=xs, qc=qc, qv=qv)
        assert qz.headsum.shape == (N,)
        assert qz.stabilizer.shape == (N,)
        sn_infsum = qz.headsum
        stabilizer = qz.stabilizer
        lastterm = -np.mean(stabilizer + np.log(sn_infsum), axis=0) / (D)
        self._logger.info(f"lastterm: {lastterm}")

        free_energy = kl_beta + kl_gauss + lastterm
        self._logger.info(f"free_energy: {free_energy}")

        elbo = -free_energy
        self._logger.info(f"elbo: {elbo}")
        return elbo

    def get_mean_stick_lengths(self, *, qv):
        means = qv.alpha / (qv.alpha + qv.beta)  # [T]
        minus = 1 - means  # [T]
        minus_prod = np.cumprod(minus, axis=0)  # [T]
        minus_prod = np.pad(minus_prod[0:-1], ((1, 0)), mode='constant', constant_values=1.0)
        return means * minus_prod

    def permute_cluster_ids(self, *, qc, qv, qz):
        stick_means = self.get_mean_stick_lengths(qv=qv)
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

    def print_stick_lengths(self, *, qv):
        stick_means = self.get_mean_stick_lengths(qv=qv)
        for i in range(qv.alpha.shape[0]):
            print(stick_means[i])

    def run(self):
        pv = self.get_pv()
        pc = self.get_pc()
        qv = self.get_qv_initial()
        qc = self.get_qc_initial()
        qc_snapshots = [qc]
        sticklen_snapshots = [self.get_mean_stick_lengths(qv=qv)]

        qz = self.update_qz(xs=self._xs, qc=qc, qv=qv)
        elbo = self.get_elbo_normalized(xs=self._xs, qc=qc, qv=qv, pc=pc, pv=pv)
        print(f"ELBO normalized: {elbo}")

        for _ in range(0, 10):
            # qc, qv, qz = self.permute_cluster_ids(qc=qc, qv=qv, qz=qz)
            qc = self.update_qc(qc=qc, xs=self._xs, qz=qz, kappa=self._kappa)
            qv = self.update_qv(qv=qv, qz=qz, pv=pv, kappa=self._kappa)
            qz = self.update_qz(xs=self._xs, qc=qc, qv=qv)
            elbo = self.get_elbo_normalized(xs=self._xs, qc=qc, qv=qv, pc=pc, pv=pv)
            print(f"ELBO normalized: {elbo}")
            qc_snapshots.append(qc)
            sticklen_snapshots.append(self.get_mean_stick_lengths(qv=qv))

        return dict(
            qc=qc, 
            qv=qv, 
            qz=qz, 
            qc_snapshots=qc_snapshots, 
            sticklen_snapshots=sticklen_snapshots,
        )

    def streamlit_demo(self, *, xs, qc_snapshots, sticklen_snapshots, **kwargs):
        dfs = []
        for i in range(self._truncation_level):
            df = pd.DataFrame({
                "timestep": range(11), 
                "x": [qc.mean[i][0] for qc in qc_snapshots], 
                "y": [qc.mean[i][1] for qc in qc_snapshots],
                "size": [1000 * sl[i] for sl in sticklen_snapshots],
            })
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        st.set_page_config(page_title="Simple VDP: Cluster Fitting in 2D", layout="centered")
        st.title("Simple VDP: Cluster Fitting in 2D")
        st.write("Use the slider to change the time step.")
        current_step = st.slider(
            "Select Time Step", min_value=0, max_value=10, value=0, step=1
        )

        filtered_df = df[df["timestep"] == current_step]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered_df["x"],
                y=filtered_df["y"],
                mode="markers",
                name="Clusters",
                marker=dict(
                    size=filtered_df["size"], 
                    sizemode="area",                 
                    sizemin=4                        
                )
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs[:, 0],
                y=xs[:, 1],
                mode="markers",
                name="Datapoints",
                opacity=0.2,
            )
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    np.random.seed(42)
    xs0 = np.random.normal(loc=0.5, scale=0.05, size=[NUM_DATA // 2, DIM_DATA])
    xs1 = np.random.normal(loc=-0.5, scale=0.05, size=[NUM_DATA // 2, DIM_DATA])
    xs = np.concatenate([xs0, xs1], axis=0)
    simple_vi = SimpleVI(
        truncation_level=TRUNCATION_LEVEL, 
        sigma_c=SIGMA_C,
        sigma_x=SIGMA_X,
        xs=xs,
        kappa=1.0,
    )
    returns = simple_vi.run()
    simple_vi.print_stick_lengths(qv=returns["qv"])
    simple_vi.streamlit_demo(xs=xs, **returns)
