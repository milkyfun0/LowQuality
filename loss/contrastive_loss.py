from __future__ import print_function

import torch
import torch.nn as nn


class Contrastive_Loss(nn.Module):
    def __init__(self, temp=0.7):
        super(Contrastive_Loss, self).__init__()
        self.temp = temp

    def forward(self, features, labels):
        loss = calc_contrastive_loss(features, features, labels, labels)
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        #
        # batch_size = features.shape[0]
        # labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float().to(device)
        # contrast_feature = features
        #
        # cosine = torch.div(torch.matmul(contrast_feature, features.T), self.temp)
        #
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * 1).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask
        #
        # logits = torch.exp(cosine) * logits_mask
        # e = torch.log(torch.exp(cosine))
        # log_prob = e - torch.log(logits.sum(1, keepdim=True))
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-3)
        #
        # loss = - mean_log_prob_pos.mean()

        return loss


def calc_contrastive_loss(
        feature_row: torch.Tensor,
        feature_col: torch.Tensor,
        label_row: torch.Tensor,
        label_col: torch.Tensor,
        mask_diag: bool = False,
        div: int = 1,
        t: float = 0.1,
        eps: float = 1e-10
):
    """
    contrastive_loss
    :param div:
    :param feature_row: (b, n)
    :param feature_col: (b, n)
    :param label_row: (b, -1)
    :param label_col: (b, -1)
    :param mask_diag:  True: diag is not pair, False: diag is not pair
    :param t: temperature
    :param eps:
    :return:
    """
    assert feature_row.shape == feature_col.shape

    label_col = torch.div(label_col, div, rounding_mode="floor")

    feature_row = feature_row / feature_row.norm(dim=1, keepdim=True)
    feature_col = feature_col / feature_col.norm(dim=1, keepdim=True)
    # print(label_row.reshape(-1, 1) == label_col.reshape(1, -1))
    mask = (label_row.reshape(-1, 1) == label_col.reshape(1, -1)).to(torch.int32)

    mask_diag = (1 - torch.eye(feature_row.shape[0], device=feature_row.device)) if mask_diag else torch.ones_like(
        mask, device=feature_row.device)

    mask = mask * mask_diag
    row_col = feature_row @ feature_col.T / t * mask_diag
    col_row = feature_col @ feature_row.T / t * mask_diag

    row_col_loss = calc_contrastive_loss_part(sim=row_col, mask=mask, eps=eps)
    col_row_loss = calc_contrastive_loss_part(sim=col_row, mask=mask.T, eps=eps)

    return (row_col_loss + col_row_loss) / 2


def calc_contrastive_loss_part(
        sim: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-10
):
    """
    :param sim: (b, b)
    :param mask: (b, b)
    :param eps:
    :return:
    """

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    sim = sim * mask

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    mask_sum = mask.sum(dim=1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

    loss = -1 * (mask * log_prob).sum(dim=1) / mask_sum.detach()
    return loss.mean()
