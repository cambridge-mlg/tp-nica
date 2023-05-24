from utils import matching_sources_corr

from sklearn.decomposition import FastICA
import pdb

def linearICA_eval(x, s_gt):
    N = s_gt.shape[1]
    M = x.shape[1]
    ica = FastICA(n_components=N)
    x = x.swapaxes(1, 2).reshape(-1, M)
    s_gt = s_gt.swapaxes(1, 2).reshape(-1, N)
    s_est = ica.fit_transform(x)

    # evaluate
    mcc, _, sort_idx = matching_sources_corr(s_est.T, s_gt.T)
    return s_est, mcc
