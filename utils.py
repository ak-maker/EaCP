"""
Here we implement some CP / training helper functions.
"""
import numpy as np
import torch.nn
import torch

from conformal import conformal_prediction as cp
from uncertainty_functions import smx_entropy
from conformal import evaluation


def pinball_loss_grad(y: float, yhat: np.ndarray, q: float) -> np.ndarray:
    """
    Compute the gradient of the pinball loss function.

    :param y: True values.
    :param yhat: Predicted values.
    :param q: Quantile level for the loss calculation.
    
    :return: Gradient of the pinball loss.
    """
    return -q * (y > yhat) + (1 - q) * (y < yhat)


def split_conformal(results: list[dict],
                    cal_path: str,
                    alpha: float,
                    cp_method: str) -> tuple[list[dict], float]:
    """
    Perform split conformal prediction and obtain the conformal threshold.

    :param results: List of dicts that will store metrics of interest.
    :param cal_path: Path to saved softmax outputs / labels from the calibration dataset.
    :param alpha: Target error level for conformal prediction; coverage is 1 - alpha.
    :param cp_method: Which cp method to use
    :param cp_method: The conformal prediction method to use ('thr' or 'raps').
    
    :return: Updated results list and conformal threshold (tau_thr).
    """
    # # # # # # # # CALIBRATION # # # # # # #
    print('Calibrating conformal')

    # start by loading and calibrating on imagenet1k validation set
    data = np.load(cal_path)
    smx = data['smx']  # get softmax scores
    labels = data['labels'].astype(int)

    # Split the softmax scores into calibration and validation sets
    n = int(len(labels) * 0.5)
    idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx, :], smx[~idx, :]
    cal_labels, val_labels = labels[idx], labels[~idx]

    # evaluate accuracy
    acc_cal = evaluation.compute_accuracy(val_smx, val_labels)
    # calibrate on imagenet calibration set
    if cp_method == 'thr':
        tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)  # get conformal quantile
    elif cp_method == 'raps':
        tau_thr = cp.calibrate_raps(cal_smx, cal_labels, alpha, k_reg=5, lambda_reg=0.01, rng=True)
    else:
        raise ValueError('CP method not supported choose from [thr, raps]')

    # get confidence sets
    if cp_method == 'thr':
        conf_set_thr = cp.predict_threshold(val_smx, tau_thr)
    elif cp_method == 'raps':
        conf_set_thr = cp.predict_raps(val_smx, tau_thr, k_reg=5, lambda_reg=0.01, rng=True)
    else:
        raise ValueError('CP method not supported choose from [thr, raps]')

    # evaluate coverage
    cov_thr_in1k = float(evaluation.compute_coverage(conf_set_thr, val_labels))
    # evaluate set size
    size_thr_in1k, _ = evaluation.compute_size(conf_set_thr)
    print(f'Accuracy on Calibration data: {acc_cal}')
    print(f'Coverage on Calibration data: {cov_thr_in1k}')
    print(f'Inefficiency on Calibration data: {size_thr_in1k}')

    results_dict = {
        'update': 'calibration',
        'cal_acc': acc_cal,
        'cal_cov': cov_thr_in1k,
        'cal_size': size_thr_in1k
    }
    results.append(results_dict)

    # ===== NEW: Compute calibration entropy stats for adaptive beta =====
    cal_ent = -(smx * np.log(smx + 1e-10)).sum(1)
    cal_ent_std = float(np.std(cal_ent))
    # ===== END NEW =====

    return results, tau_thr, cal_ent_std


def update_beta_online(output_ent: torch.Tensor, beta: float, alpha: float) -> float:
    """
    Update the estimated \beta entropy quantile online for use in adapting the conformal prediction threshold, see
    Eq. 3 of the paper.

    :param output_ent: Entropy of the output predictions.
    :param beta: Entropy quantile estimate.
    :param alpha: Target error level (1 - alpha = coverage).
    
    :return: Updated entropy quantile.
    """
    # update the beta entropy quantile using pinball loss
    loss = pinball_loss_grad(beta, output_ent.cpu().detach().numpy(), alpha).mean()
    beta += loss

    return beta


def update_beta_batch(output_ent: torch.Tensor, alpha: float) -> float:
    """
    Instead of updating the \beta quantile online, we can simply use the entropy quantile on a particular batch of data
    (or the entire dataset if available). On a large enough batch size, the difference with online estimate is
    negligible.

    :param output_ent: Entropy of the output predictions.
    :param alpha: Target error level (1 - alpha = coverage).
    :return: Entropy quantile of the batch / dataset.
    """

    # Find the entropy quantile on the batch of data
    upper_q = np.quantile(output_ent.cpu().detach().numpy(), 1 - alpha)

    return upper_q


# ===== NEW: Adaptive beta based on entropy variance =====
def update_beta_adaptive(output_ent: torch.Tensor, alpha: float, cal_ent_std: float) -> float:
    """
    Adaptive beta: adjust the quantile level based on how much the test entropy
    distribution deviates from calibration. Higher variance (more shift) leads to
    a higher quantile, producing a larger beta for more aggressive scaling.

    :param output_ent: Entropy (or other uncertainty) of the output predictions.
    :param alpha: Target error level (1 - alpha = coverage).
    :param cal_ent_std: Standard deviation of entropy on the calibration set.
    :return: Adapted entropy quantile.
    """
    ent = output_ent.cpu().detach().numpy()
    test_std = np.std(ent)
    # shift_ratio > 1 means test entropy is more spread out than calibration
    shift_ratio = test_std / (cal_ent_std + 1e-8)
    # When shift is severe, reduce alpha -> higher quantile -> larger beta
    adjusted_alpha = alpha / max(shift_ratio, 1.0)
    adjusted_alpha = max(adjusted_alpha, 0.01)  # cap at 99th percentile
    upper_q = np.quantile(ent, 1 - adjusted_alpha)
    return upper_q
# ===== END NEW =====


# ===== NEW: Sliding window beta =====
class SlidingWindowBeta:
    """Maintain a sliding window of recent entropy values for more stable beta estimation."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = []

    def update(self, output_ent: torch.Tensor, alpha: float) -> float:
        ent = output_ent.cpu().detach().numpy()
        self.buffer.append(ent)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        all_ent = np.concatenate(self.buffer)
        return float(np.quantile(all_ent, 1 - alpha))


def compute_adaptive_scaling(output_ent: torch.Tensor, cal_ent_std: float,
                             base_s: int = 2) -> float:
    """Adaptive scaling factor: adjust s based on detected shift magnitude.
    Small shift -> smaller s (less inflation), large shift -> larger s."""
    ent = output_ent.cpu().detach().numpy()
    test_std = np.std(ent)
    shift_ratio = test_std / (cal_ent_std + 1e-8)
    # scale s with sqrt of shift ratio for gentler adjustment: clamp to [1, 3]
    adaptive_s = base_s * max(shift_ratio ** 0.5, 0.5)
    adaptive_s = min(max(adaptive_s, 1.0), 3.0)
    return adaptive_s
# ===== END NEW =====


def t2sev(t, run_length=7, schedule=None):
    """
    Time step to severity level, for continious shifts.
    """
    t_base = t
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    else:
        return 5 * ((t_base // run_length) % 2)  # default: sudden schedule
