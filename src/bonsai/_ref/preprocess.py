"""Vectorised fp64 port of upstream's ``read_and_filter`` plus the √v_g rescaling (amendment A5).

Reproduces DP:2792-3135 and DP:204-230 bitwise: same elementwise operations in the same order
(clamp → z-score → mask → max(·, 1e-6) → ``*= f`` → ``+= mu`` → ``/= sqrt(v)``), applied to whole
genes × cells arrays; row sums are NumPy pairwise sums along the contiguous axis, as upstream's
per-row ``np.sum``. Upstream reads the float64 text written by ``_io_folder``; float32 layers are
widened to float64 first, which is exact, as the text round trip is.
"""

from typing import NamedTuple

import numpy as np
from anndata import AnnData
from scipy.optimize import minimize

from bonsai._core.dataprocessing import neg_loglik_grad_using_measurements_and_errors
from bonsai._core.helpers import BonsaiError
from bonsai._io_folder import validate_inputs


class Leaves(NamedTuple):
    x: np.ndarray  # (N, G) rescaled likelihood means, cells in adata order, kept genes only
    var: np.ndarray  # (N, G) rescaled likelihood variances
    genes: np.ndarray  # (n_vars,) bool, genes kept (upstream's var["bonsai_used"])
    gene_var: np.ndarray  # (G,) upstream's geneVariances of the kept genes
    loglik_var_corr: float  # upstream's loglikVarCorr (logged log-likelihoods add it; finalLoglik does not)


def _rows(a) -> np.ndarray:
    """Cells × genes layer → C-contiguous float64 genes × cells (upstream's row = one gene)."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).T)


def preprocess(
    adata: AnnData, zscore_cutoff: float = 1.0, rescale_by_var: bool = True, all_genes: bool = False
) -> Leaves:
    """Upstream's leaf arrays for ``adata`` (``uns["bonsai_input"]`` from ``pp.read_sanity``/``pp.from_arrays``).

    Raises ``BonsaiError`` (a ``ValueError``) if fewer than 2 genes are kept (DP:3083-3086).
    """
    mean, sd = validate_inputs(adata)
    if all_genes:
        zscore_cutoff = -1  # SCData.__init__ (DP:174-175)
    tau = -1 if all_genes else 1e-4  # divide_out_prior_cutoff (DP:2819-2821)
    zsq_cut = zscore_cutoff**2 if zscore_cutoff > 0 else 0.0  # DP:2869
    m = _rows(mean)
    n_cells = m.shape[1]
    if adata.uns["bonsai_input"]["kind"] == "sanity_posterior":  # DP:2900-2932
        var = np.maximum(_rows(sd) ** 2, 1e-12)
        v = np.asarray(adata.var["bonsai_prior_var"], dtype=np.float64)
        mu = np.asarray(adata.var["bonsai_mu"], dtype=np.float64)
        keep = np.sum(m**2 / var, axis=1) / n_cells > zsq_cut
        D = v[:, None] - var
        keep &= ~np.any(D < tau, axis=1)
        D = np.maximum(D[keep], 1e-6)
        f = v[keep, None] / D
        x = m[keep] * f
        var = var[keep] * f
        x += mu[keep, None]
        gene_var = v[keep]
    elif sd is not None:  # likelihood with SDs (DP:2939-2980): per-gene BFGS, as upstream
        var_all = np.maximum(_rows(sd) ** 2, 1e-12)
        keep = np.zeros(m.shape[0], dtype=bool)
        gv = np.zeros(m.shape[0])
        for g in range(m.shape[0]):
            means, vars_ = m[g], var_all[g]
            try:  # DP:2955: np.var == 0 warns (RuntimeWarning), never raises, so this never fires
                log_v0 = np.log(np.var(means))
            except RuntimeWarning:
                log_v0 = 0
            res = minimize(
                neg_loglik_grad_using_measurements_and_errors,
                x0=log_v0,
                jac=True,
                args=(means, vars_, False),
            )
            if not res.success:
                continue
            v_hat = np.exp(res.x[0])
            _, _, mu_hat = neg_loglik_grad_using_measurements_and_errors(res.x, means, vars_, return_mean_ML=True)
            keep[g] = np.sum((v_hat / (v_hat + vars_)) * (means - mu_hat) ** 2 / vars_) / n_cells > zsq_cut
            gv[g] = v_hat
        x, var, gene_var = m[keep], var_all[keep], gv[keep]
    else:  # no SDs (DP:2993-3026)
        gv = np.var(m, axis=1)
        keep = gv > 0.0
        x, gene_var = m[keep], gv[keep]
        var = np.ones(x.shape) * 1e-6
    if keep.sum() < 2:
        raise BonsaiError(
            f"{int(keep.sum())} gene(s) passed the filters; at least 2 are needed. Consider lowering the cutoff."
        )
    n_genes = x.shape[0]
    if rescale_by_var:  # DP:205-214
        x /= np.sqrt(gene_var[:, None])
        var /= gene_var[:, None]
        corr = -(n_cells - 1) * np.sum(np.log(gene_var))
    else:  # DP:221-230
        s = np.mean(gene_var)
        x /= np.sqrt(s)
        var /= s
        corr = -(n_cells - 1) * n_genes * np.log(s)
    return Leaves(x.T.copy(), var.T.copy(), keep, gene_var, float(corr))
