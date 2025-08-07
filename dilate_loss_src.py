# Contenuto CORRETTO per il file: dilate_loss_src.py

import torch
import torch.nn as nn
import numpy as np

# Funzioni helper interne (_soft_dtw e _path) non necessitano di modifiche
# perché vengono chiamate all'interno di un blocco no_grad nella funzione principale.

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
            # Questa operazione è complessa, ma non dobbiamo differenziarla direttamente
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
    
    # Calcolo della matrice di distanza tra ogni punto.
    # Questo calcolo DEVE essere differenziabile, quindi rimane fuori dal no_grad.
    D_xy = torch.zeros((B, N, M)).to(device)
    for n in range(N):
        for m in range(M):
            D_xy[:, n, m] = torch.sum((y_true[:, n, :] - y_pred[:, m, :]) ** 2, dim=1)
            
    # --- INIZIO BLOCCO CORRETTIVO ---
    # Calcoliamo il percorso ottimale e i pesi (omega) senza tracciare i gradienti,
    # perché sono considerati costanti rispetto ai parametri del modello per la loss_shape.
    with torch.no_grad():
        D_xy_nograd = D_xy.detach()
        R = _soft_dtw(gamma, D_xy_nograd, D_xy_nograd)
        omega = _path(R)
    # --- FINE BLOCCO CORRETTIVO ---
    
    # La loss di forma (loss_shape) ora usa la matrice di distanza differenziabile (D)
    # e i pesi non differenziabili (omega). Questo è il modo corretto.
    loss_shape = torch.sum(D_xy * omega, dim=(1, 2)) / (torch.sum(omega, dim=(1, 2)))

    # --- INIZIO BLOCCO CORRETTIVO PER LOSS TEMPORALE ---
    with torch.no_grad():
        T_y_true = torch.zeros(B, N, N).to(device)
        T_y_pred = torch.zeros(B, M, M).to(device)
        for b in range(B):
            T_y_true[b, :, :] = torch.cdist(y_true[b, :, :], y_true[b, :, :])
            T_y_pred[b, :, :] = torch.cdist(y_pred[b, :, :], y_pred[b, :, :])
    # --- FINE BLOCCO CORRETTIVO PER LOSS TEMPORALE ---
            
    loss_temporal = torch.mean(torch.abs(torch.bmm(torch.bmm(torch.transpose(omega, 1, 2), T_y_true), omega) - T_y_pred))
    
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    
    return loss, loss_shape, loss_temporal
