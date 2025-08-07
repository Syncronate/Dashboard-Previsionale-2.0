# Contenuto CORRETTO e STABILE per il file: dilate_loss_src.py

import torch
import torch.nn as nn
import numpy as np

def _soft_dtw_stable(gamma, D):
    """
    Calcola il soft-DTW in modo numericamente stabile usando il Log-Sum-Exp trick.
    """
    B, N, M = D.shape
    R = torch.full((B, N + 1, M + 1), float('inf')).to(D.device)
    R[:, 0, 0] = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Valori da cui calcolare il soft-min
            vals = torch.stack([R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1]])
            
            # Log-Sum-Exp trick:
            # log(sum(exp(x_i))) = m + log(sum(exp(x_i - m)))
            # dove m = max(x_i)
            # Per il soft-min, la formula diventa: -gamma * log(sum(exp(-x_i / gamma)))
            # E con il trick: -gamma * (-m/gamma + log(sum(exp(-(x_i - m)/gamma)))) = m - gamma * log(sum(exp(-(x_i - m)/gamma)))

            m, _ = torch.max(-vals / gamma, dim=0)
            soft_min = -gamma * (m + torch.log(torch.sum(torch.exp(-vals / gamma - m), dim=0)))
            
            R[:, i, j] = D[:, i - 1, j - 1] + soft_min
            
    return R

def _path_stable(R):
    """
    Calcola il percorso di allineamento atteso.
    Questa funzione è sensibile a valori molto grandi, quindi usiamo alcuni clamp.
    """
    B, N_plus_1, M_plus_1 = R.shape
    N, M = N_plus_1 - 1, M_plus_1 - 1
    E = torch.zeros_like(R)
    E[:, 0, 0] = 1

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Clamp per evitare overflow in exp
            val1 = torch.clamp((R[:, i - 1, j] - R[:, i, j]), -10, 10)
            val2 = torch.clamp((R[:, i, j - 1] - R[:, i, j]), -10, 10)
            val3 = torch.clamp((R[:, i - 1, j - 1] - R[:, i, j]), -10, 10)

            E[:, i, j] = (E[:, i-1, j] * torch.exp(val1) +
                          E[:, i, j-1] * torch.exp(val2) +
                          E[:, i-1, j-1] * torch.exp(val3))
    return E[:, 1:, 1:]


def dilate_loss(y_true, y_pred, alpha, gamma, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    B, N, _ = y_true.shape
    M = y_pred.shape[1]
    
    # Calcolo della matrice di distanza L2
    D_xy = torch.cdist(y_true, y_pred, p=2.0).pow(2)

    # Calcolo del soft-DTW e del percorso, ora con le funzioni stabili
    with torch.no_grad():
        R = _soft_dtw_stable(gamma, D_xy.detach())
        omega = _path_stable(R)
    
    # Calcolo della loss di forma (shape)
    loss_shape = torch.sum(D_xy * omega, dim=(1, 2)) / N

    # Calcolo della loss temporale
    with torch.no_grad():
        # Matrici di distanza temporale (distanza tra i frame nella stessa serie)
        T_y_true = torch.cdist(y_true, y_true, p=2.0).pow(2)
        T_y_pred = torch.cdist(y_pred, y_pred, p=2.0).pow(2)

    loss_temporal = torch.mean(torch.abs(torch.bmm(torch.bmm(torch.transpose(omega, 1, 2), T_y_true), omega) - T_y_pred))

    # Loss finale combinata
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    
    return loss, loss_shape, loss_temporal
