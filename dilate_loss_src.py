import torch
import torch.nn as nn
import numpy as np

def _soft_dtw(gamma, D, D_xy):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    D_xy = D_xy.view(B, N, M)
    R = torch.zeros((B, N + 2, M + 2)).to(D.device) + 1e8
    R[:, 1, 1] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            r0 = R[:, i, j]
            r1 = R[:, i - 1, j]
            r2 = R[:, i, j - 1]
            r3 = R[:, i - 1, j - 1]
            R[:, i, j] = D[:, i - 1, j - 1] - gamma * torch.log(
                torch.exp((r1 - r0) / gamma) + torch.exp((r2 - r0) / gamma) + torch.exp((r3 - r0) / gamma)
            )
    return R

def _path(R):
    B = R.shape[0]
    N = R.shape[1] - 2
    M = R.shape[2] - 2
    E = torch.zeros((B, N + 2, M + 2)).to(R.device)
    E[:, 1, 1] = 1
    R_ = (R < 1e6).float() * R
    for i in range(2, N + 2):
        E[:, i, 1] = torch.exp((R_[:, i - 1, 1] - R_[:, i, 1])) * E[:, i - 1, 1]
    for j in range(2, M + 2):
        E[:, 1, j] = torch.exp((R_[:, 1, j - 1] - R_[:, 1, j])) * E[:, 1, j - 1]
    for i in range(2, N + 2):
        for j in range(2, M + 2):
            e0 = E[:, i, j - 1] * torch.exp((R_[:, i, j - 1] - R_[:, i, j]))
            e1 = E[:, i - 1, j] * torch.exp((R_[:, i - 1, j] - R_[:, i, j]))
            e2 = E[:, i - 1, j - 1] * torch.exp((R_[:, i - 1, j - 1] - R_[:, i, j]))
            E[:, i, j] = e0 + e1 + e2
    return E[:, 1:-1, 1:-1]

def dilate_loss(y_true, y_pred, alpha, gamma, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    B = y_true.shape[0]
    N = y_true.shape[1]
    M = y_pred.shape[1]
    
    D_xy = torch.zeros((B, N, M)).to(device)
    for n in range(N):
        for m in range(M):
            D_xy[:, n, m] = torch.sum((y_true[:, n, :] - y_pred[:, m, :]) ** 2, dim=1)
    
    D = D_xy.view(B, N, M)
    R = _soft_dtw(gamma, D, D_xy)
    R = R.to(device)
    gamma = torch.tensor(gamma).to(device)
    
    R_diag = torch.zeros(B, N, M).to(device)
    for n in range(N):
        for m in range(M):
            R_diag[:, n, m] = R[:, n, m + 1] + R[:, n + 1, m] - R[:, n, m] - R[:, n + 1, m + 1]
            
    R_diag = (R_diag / (2 * gamma))
    
    omega = _path(R)
    
    loss_shape = torch.sum(D * omega, dim=(1, 2)) / (torch.sum(omega, dim=(1, 2)))
    
    D_t = torch.bmm(torch.transpose(y_true, 1, 2), y_true)
    
    T_y_true = torch.zeros(B, N, N).to(device)
    T_y_pred = torch.zeros(B, M, M).to(device)
    
    for b in range(B):
        T_y_true[b, :, :] = torch.cdist(torch.transpose(y_true, 1, 2)[b, :, :].view(N, -1), torch.transpose(y_true, 1, 2)[b, :, :].view(N, -1))
        T_y_pred[b, :, :] = torch.cdist(torch.transpose(y_pred, 1, 2)[b, :, :].view(M, -1), torch.transpose(y_pred, 1, 2)[b, :, :].view(M, -1))
        
    T_y_true_ = T_y_true.view(B, N, N)
    T_y_pred_ = T_y_pred.view(B, M, M)
    
    om_T = torch.bmm(omega, torch.transpose(omega, 1, 2))
    
    K_T = T_y_true_
    K_T_ = T_y_pred_
    
    loss_temporal = torch.sum(torch.abs(torch.bmm(torch.bmm(torch.transpose(omega, 1, 2), K_T), omega) - K_T_))
    
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    
    return loss, loss_shape, loss_temporal
