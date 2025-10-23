# -*- coding: utf-8 -*-
import torch

class TorchPLS:
    def __init__(self, n_components=2, device=None, ridge=1e-8):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_components = int(n_components)
        self.device = device
        self.ridge = float(ridge)

        self.X_mean = None
        self.Y_mean = None
        self.W = None
        self.P = None
        self.Q = None
        self.B = None
        self.n_targets_ = None

    def _to_2d_y(self, Y):
        if Y.dim() == 1:
            Y = Y.unsqueeze(1)
        return Y

    def fit(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        Y = self._to_2d_y(Y)

        if X.dim() != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got {tuple(X.shape)}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same n_samples: {X.shape[0]} vs {Y.shape[0]}")

        # Fail fast if non-finite in raw inputs
        if not torch.isfinite(X).all():
            bad = (~torch.isfinite(X)).sum().item()
            raise ValueError(f"Non-finite values in X (count={bad}).")
        if not torch.isfinite(Y).all():
            bad = (~torch.isfinite(Y)).sum().item()
            raise ValueError(f"Non-finite values in Y (count={bad}).")

        n_samples, n_features = X.shape
        _, n_targets = Y.shape
        self.n_targets_ = int(n_targets)

        # Center
        self.X_mean = X.mean(0)
        self.Y_mean = Y.mean(0)
        Xc = X - self.X_mean
        Yc = Y - self.Y_mean

        T_list, P_list, Q_list, W_list = [], [], [], []

        for _ in range(self.n_components):
            # w ~ sum over targets of cov(X, Y)
            w = (Xc.T @ Yc).sum(dim=1, keepdim=True)  # (n_features, 1)

            # If w is zero vector, stop adding components
            w_norm = torch.norm(w)
            if not torch.isfinite(w_norm) or w_norm.item() == 0.0:
                break
            w = w / (w_norm + 1e-12)

            t = Xc @ w  # (n_samples, 1)
            denom = (t.T @ t)  # 1x1

            # If denom ~ 0 or non-finite, stop
            if (not torch.isfinite(denom).all()) or torch.abs(denom).item() < 1e-12:
                break

            q = (Yc.T @ t) / denom        # (n_targets,  1)
            p = (Xc.T @ t) / denom        # (n_features, 1)

            # Deflate
            Xc = Xc - t @ p.T
            Yc = Yc - t @ q.T

            # Any non-finite after deflation? Bail cleanly.
            if (not torch.isfinite(Xc).all()) or (not torch.isfinite(Yc).all()):
                raise ValueError("Non-finite produced during deflation step.")

            T_list.append(t)
            P_list.append(p)
            Q_list.append(q)
            W_list.append(w)

        if not W_list:
            raise RuntimeError("Failed to extract any PLS components (degenerate data / all-zero variance).")

        self.W = torch.cat(W_list, dim=1)  # (n_features, n_comp_eff)
        self.P = torch.cat(P_list, dim=1)  # (n_features, n_comp_eff)
        self.Q = torch.cat(Q_list, dim=1)  # (n_targets,  n_comp_eff)

        PtW = self.P.T @ self.W            # (n_comp, n_comp)

        # Tiny ridge to improve SVD stability
        eye = torch.eye(PtW.shape[0], dtype=PtW.dtype, device=PtW.device)
        PtW_reg = PtW + self.ridge * eye

        if not torch.isfinite(PtW_reg).all():
            raise ValueError("Non-finite values in (P^T W) before pinv.")

        self.B = self.W @ torch.linalg.pinv(PtW_reg) @ self.Q.T  # (n_features, n_targets)
        return self

    def predict(self, X):
        if self.B is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if X.dim() != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features); got {tuple(X.shape)}")

        Xc = X - self.X_mean
        Y_pred = Xc @ self.B + self.Y_mean  # (n_samples, n_targets)
        out = Y_pred.detach().cpu().numpy()
        return out.ravel() if self.n_targets_ == 1 else out
