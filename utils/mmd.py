import torch
import torch.nn as nn


def MMD_batch(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P,  shape = (batch_size, sample_size_1, d)
        y: second sample, distribution Q, shape = (batch_size, sample_size_2, d)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    bs = x.shape[0]

    xx = x @ x.transpose(1, 2)  # shape = (batch_size, sample_size_1, sample_size_1)
    yy = y @ y.transpose(1, 2)  # shape = (batch_size, sample_size_2, sample_size_2)
    zz = x @ y.transpose(1, 2)  # shape = (batch_size, , sample_size_1, sample_size_2)

    rx = (
        xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
    )  # (batch_size, sample_size_1, sample_size_1)
    ry = (
        yy.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(yy)
    )  # (batch_size, sample_size_1, sample_size_1)

    dxx = rx.transpose(1, 2) + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.transpose(1, 2) + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.transpose(1, 2) + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return (XX + YY - 2.0 * XY).view(bs, -1).mean(1)


def sliding_window_mmd_batch(
    ensemble_preds, window_size, kernel="rbf", step: int = 1
) -> torch.Tensor:
    _, batch_size, seq_len = ensemble_preds.shape

    time_range = torch.arange(seq_len)
    start_pos = time_range[0 : seq_len - window_size + 1 : step]
    idxs_list = [time_range[s : s + window_size] for s in start_pos]

    # first window_size elements are zeros
    mmd_scores_batch = torch.zeros((batch_size, seq_len))

    for i, (history_idx, future_idx) in enumerate(zip(idxs_list[:-1], idxs_list[1:])):
        X = ensemble_preds[:, :, history_idx].transpose(0, 1).reshape(batch_size, -1, 1)
        Y = ensemble_preds[:, :, future_idx].transpose(0, 1).reshape(batch_size, -1, 1)
        mmd_batch = MMD_batch(X, Y, kernel=kernel)

        mmd_scores_batch[:, window_size + i] = mmd_batch

    return mmd_scores_batch

def anchor_window_detector_batch(
    ensemble_preds, window_size, distance="mmd", kernel="rbf", anchor_window_type="start"
) -> torch.Tensor:
    #ensemble_preds - output of .predict_all_model(), shape is (n_models, batch_size, seq_len)
    
    assert distance in ["mmd", "cosine"], "Unknown distance type"
    assert anchor_window_type in ["start", "prev", "combined"], "Unknown window type"
    
    _, batch_size, seq_len = ensemble_preds.shape
    
    future_idx_range = torch.arange(seq_len)[window_size:]

    # first window_size elements are zeros
    dist_scores_batch = torch.zeros((batch_size, seq_len))
    
    for future_idx in future_idx_range:
        if anchor_window_type == "start":
            anchor_wnd = ensemble_preds[:, :, :window_size]
        elif anchor_window_type == "prev":
            anchor_wnd = ensemble_preds[:, :, future_idx - window_size : future_idx]
        else:
            anchor_wnd = torch.cat(
                (ensemble_preds[:, :, :window_size], ensemble_preds[:, :, future_idx - window_size : future_idx]),
                dim=-1
            )
            
        future = ensemble_preds[:, :, future_idx].transpose(0, 1).reshape(batch_size, -1, 1)
        
        dist_batch = torch.zeros(batch_size)
        # compute mean pairwise distance
        for i in range(window_size):
            anchor = anchor_wnd[:, :, i].transpose(0, 1).reshape(batch_size, -1, 1)
            
            if distance == "mmd":
                dist_batch += MMD_batch(anchor, future, kernel=kernel)
            else:
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                curr_dist = cos(anchor.squeeze(), future.squeeze())
                curr_dist = 1.0 - torch.where(curr_dist >= 0, curr_dist, torch.zeros_like(curr_dist))
                dist_batch += curr_dist
        
        dist_batch /= anchor_wnd.shape[-1]
            
        dist_scores_batch[:, future_idx] = dist_batch

    return dist_scores_batch

'''
# code from: https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P,  shape = (batch_size, sample_size, d)
        y: second sample, distribution Q, shape = (batch_size, sample_size, d)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return (XX + YY - 2.0 * XY).mean()


# my sliding window procedure
def sliding_window_mmd(ensemble_preds, window_size, kernel="rbf"):
    _, seq_len = ensemble_preds.shape

    time_range = torch.arange(seq_len)
    start_pos = time_range[0 : seq_len - window_size + 1 : 1]
    idxs_list = [time_range[s : s + window_size] for s in start_pos]

    # first window_size elements are zeros
    mmd_scores = torch.zeros(seq_len)

    for i, (history_idx, future_idx) in enumerate(zip(idxs_list[:-1], idxs_list[1:])):
        X = ensemble_preds[:, history_idx].reshape(-1, 1)
        Y = ensemble_preds[:, future_idx].reshape(-1, 1)

        mmd = MMD(X, Y, kernel=kernel)
        mmd_scores[window_size + i] = mmd

    return mmd_scores
'''