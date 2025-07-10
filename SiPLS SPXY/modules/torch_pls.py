# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:00:09 2025

@author: spect
"""

import torch

class TorchPLS:
    def __init__(self, n_components=2, device='cuda'):
        self.n_components = n_components
        self.device = device

    def fit(self, X, Y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        X_mean = X.mean(0)
        Y_mean = Y.mean(0)
        self.X_mean, self.Y_mean = X_mean, Y_mean

        Xc = X - X_mean
        Yc = Y - Y_mean

        n = X.shape[0]
        T, P, Q, W = [], [], [], []

        for _ in range(self.n_components):
            w = (Xc.T @ Yc).sum(1).unsqueeze(1)
            w = w / torch.norm(w)
            t = Xc @ w
            q = (Yc.T @ t) / (t.T @ t)
            p = (Xc.T @ t) / (t.T @ t)

            Xc = Xc - t @ p.T
            Yc = Yc - t @ q.T

            T.append(t)
            P.append(p)
            Q.append(q)
            W.append(w)

        self.W = torch.cat(W, dim=1)
        self.P = torch.cat(P, dim=1)
        self.Q = torch.cat(Q, dim=1)
        self.B = self.W @ torch.inverse(self.P.T @ self.W) @ self.Q.T

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Xc = X - self.X_mean
        Y_pred = Xc @ self.B + self.Y_mean
        return Y_pred.cpu().numpy()
