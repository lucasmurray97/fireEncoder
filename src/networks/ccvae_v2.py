import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
import re, hashlib


class CCVAE_V2(nn.Module):
    def __init__(self, params):
        super(CCVAE_V2, self).__init__()
        self.name = "CCVAE_V2"
        self.instance= params["instance"]
        self.latent_dims = params["latent_dims"]
        self.c = params["capacity"]
        self.input_size = params["input_size"]
        self.distribution_std = params["distribution_std"]
        kernel_size = 4; stride = 2; padding = 1
        self.dim_1 = int((self.input_size - kernel_size + 2*padding)/2 + 1)
        self.dim_2 = int((self.dim_1 - kernel_size + 2*padding)/2 + 1)
        self.is_sigmoid = params["sigmoid"]
        self.lr1 = params["lr1"]; self.lr2 = params["lr2"]
        self.not_reduced = params["not_reduced"]
        self.variational_beta = params["variational_beta"]
        use_gpu = params["use_gpu"]
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if use_gpu else 'cpu'
        self.latent_portion = params["latent_portion"]
        self.rec_dim  = int(self.latent_dims * (1 - self.latent_portion))
        self.burn_dim = int(self.latent_dims * self.latent_portion)
        self.alpha = params["alpha"]
        self.params = params

        # === NEW: predictor options ===
        self.use_rank_loss = bool(params.get("use_rank_loss", True))
        self.lambda_rank = float(params.get("lambda_rank", 1.0))
        self.lambda_cons = float(params.get("lambda_cons", 0.10))
        self.lambda_reg  = float(params.get("lambda_reg",  0.00))  # optional small MSE calibration
        self.jitter_std  = float(params.get("jitter_std",  0.03))

        # ---------------- Encoder ----------------
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc_mu = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.latent_dims*(self.dim_2**2), out_features = self.latent_dims)
        self.bn1 = nn.BatchNorm2d(self.c);  self.drop1 = nn.Dropout()
        self.bn2 = nn.BatchNorm2d(self.c*2); self.drop2 = nn.Dropout()
        self.encoder = nn.Sequential(
            self.conv1, self.bn1, self.drop1, nn.ReLU(),
            self.conv2, self.bn2, self.drop2, nn.ReLU(),
            nn.Flatten()
        )
        self.encoder_params = list(self.encoder.parameters()) + list(self.fc_mu.parameters()) + list(self.fc_logvar.parameters())

        # ---------------- Decoder ----------------
        self.fc = nn.Linear(in_features=self.latent_dims, out_features=self.latent_dims*(self.dim_2**2))
        self.conv1_ = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2_ = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1_2 = nn.BatchNorm2d(self.c); self.drop1_2 = nn.Dropout()
        self.decoder = nn.Sequential(
            self.fc, nn.ReLU(),
            nn.Unflatten(1, (self.c*2, self.dim_2, self.dim_2)),
            self.conv1_, self.bn1_2, self.drop1_2, nn.ReLU(),
            self.conv2_
        )

        # ---------------- Burned % predictor (LayerNorm MLP) ----------------
        self.burned_predictor = nn.Sequential(
            nn.Linear(self.burn_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        # NOTE: we'll output RAW score if use_rank_loss=True, else sigmoid(prob)

        # Kaiming init
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1_.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2_.weight, mode='fan_in', nonlinearity='relu')

        # Parameter groups / optimizers
        self.vae_params = list(self.encoder_params) + list(self.decoder.parameters())
        self.r_params = list(self.burned_predictor.parameters())
        self.optimizer_1 = torch.optim.Adam(self.vae_params, lr = self.lr1)
        self.optimizer_2 = torch.optim.Adam(self.r_params,  lr = self.lr2)
        self.criterion_vae = nn.BCELoss() if self.is_sigmoid else nn.MSELoss()
        self.criterion_r   = nn.MSELoss()

        # Logs
        self.training_loss = []; self.validation_loss = []
        self.reconstruction_training_loss = []; self.reconstruction_validation_loss = []
        self.divergence_training_loss = []; self.divergence_validation_loss = []
        self.burned_training_loss = []; self.burned_validation_loss = []
        self.spearman_training = []; self.spearman_validation = []
        self.gradnorm_training = [];  self.gradnorm_validation = []
                # ---- Structure metrics time series (appended once per epoch) ----
        self.struct_pred_spearman = []
        self.struct_ridge_spearman = []
        self.struct_ridge_R2 = []
        self.struct_lipschitz_med = []
        self.struct_lipschitz_p90 = []
        self.struct_topo = []
        self.struct_moransI = []
        self.struct_aSigmaa_free = []
        self.struct_best_gain32 = []


        self.m = self.n = 0
        self.epoch_loss = self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = self.val_divergence_epoch_loss = 0
        self.burned_epoch_loss = self.val_burned_epoch_loss = 0

        self.spearman_epoch_sum = self.val_spearman_epoch_sum = 0.0
        self.spearman_epoch_cnt = self.val_spearman_epoch_cnt = 0
        self.gradnorm_epoch_sum = self.val_gradnorm_epoch_sum = 0.0
        self.gradnorm_epoch_cnt = self.val_gradnorm_epoch_cnt = 0

        self.last_mu = None; self.last_logvar = None; self.last_latent = None

    # ---------------- Core ops ----------------
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = self.decoder(z)
        return torch.sigmoid(x) if self.is_sigmoid else x

    def _predict_burned_raw(self, z_full):
        z_free = z_full[:, self.rec_dim:]
        return self.burned_predictor(z_free).squeeze(-1)

    def predict_burned(self, z_full):
        raw = self._predict_burned_raw(z_full)
        return torch.sigmoid(raw)

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(torch.zeros_like(std), self.distribution_std).to(self.device)
            return mu + eps * std
        return mu

    def forward(self, x, _r_unused):
        mu, logvar = self.encode(x)
        z = self.latent_sample(mu, logvar)
        x_recon = self.decode(z)
        r_pred = self.predict_burned(z)   # raw if rank loss, sigmoid(prob) if not
        self.last_mu, self.last_logvar, self.last_latent = mu, logvar, z
        return x_recon, r_pred


    @staticmethod
    def _batch_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
        aa = a.detach().flatten().cpu().numpy()
        bb = b.detach().flatten().cpu().numpy()
        rho, _ = spearmanr(aa, bb)
        return float(0.0 if np.isnan(rho) else rho)

    def vae_loss(self, recon_x, x, mu, logvar, pred_r, r_true):
        # recon
        if self.is_sigmoid:
            if not self.not_reduced:
                recon_loss = F.binary_cross_entropy(
                    recon_x.view(-1, 400), x[:, 0, :, :].view(-1, 400), reduction='mean'
                )
            else:
                recon_loss = self.criterion_vae(recon_x, x[:, 0, :, :])
        else:
            recon_loss = self.criterion_vae(recon_x, x[:, 0, :, :])

        # KL per dim per sample
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)   # [B, D]
        kldivergence = kl_per_dim.sum(dim=1).mean()                   # scalar: nats / sample

        eps = 1e-6
        r_logit = torch.logit(torch.clamp(r_true.view(-1), eps, 1 - eps))
        raw_r = torch.logit(torch.clamp(pred_r.view(-1), eps, 1 - eps))
        mse_logit = F.mse_loss(raw_r.view(-1), r_logit)
        pred_total = self.alpha * mse_logit

        total = recon_loss + self.variational_beta * kldivergence + pred_total
        return recon_loss, kldivergence, pred_total, total

    # ---------------- Hooks used by your loop ----------------
    def loss(self, output, x, r):
        x_rec, r_pred = output
        mu, logvar = self.last_mu, self.last_logvar
        l1, l2, l3, L = self.vae_loss(x_rec, x, mu, logvar, r_pred, r)
        self.reconstruction_epoch_loss += l1.item()
        self.divergence_epoch_loss     += l2.item()
        self.burned_epoch_loss         += l3.item()
        self.epoch_loss                += L.item()
        self.n += 1

        # metrics
        rho = self._batch_spearman(r_pred, r)
        self.spearman_epoch_sum += rho; self.spearman_epoch_cnt += 1
        try:
            s_sum = (r_pred if self.use_rank_loss else torch.logit(torch.clamp(r_pred, 1e-6, 1-1e-6))).sum()
            g = torch.autograd.grad(s_sum, self.last_latent, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                g_free = g[:, self.rec_dim:]; gn = g_free.norm(dim=1).mean().item()
                self.gradnorm_epoch_sum += gn; self.gradnorm_epoch_cnt += 1
        except Exception:
            pass
        return L

    def val_loss(self, output, x, r):
        x_rec, r_pred = output
        mu, logvar = self.last_mu, self.last_logvar
        l1, l2, l3, L = self.vae_loss(x_rec, x, mu, logvar, r_pred, r)
        self.val_reconstruction_epoch_loss += l1.item()
        self.val_divergence_epoch_loss     += l2.item()
        self.val_burned_epoch_loss         += l3.item()
        self.val_epoch_loss                += L.item()
        self.m += 1

        rho = self._batch_spearman(r_pred, r)
        self.val_spearman_epoch_sum += rho; self.val_spearman_epoch_cnt += 1
        try:
            s_sum = (r_pred if self.use_rank_loss else torch.logit(torch.clamp(r_pred, 1e-6, 1-1e-6))).sum()
            g = torch.autograd.grad(s_sum, self.last_latent, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                g_free = g[:, self.rec_dim:]; gn = g_free.norm(dim=1).mean().item()
                self.val_gradnorm_epoch_sum += gn; self.val_gradnorm_epoch_cnt += 1
        except Exception:
            pass
        return L

    def step(self):
        self.optimizer_1.step(); self.optimizer_2.step()

    def zero_grad(self):
        self.optimizer_1.zero_grad(); self.optimizer_2.zero_grad()

    def reset_losses(self):
        self.training_loss.append(self.epoch_loss/max(1,self.n))
        self.validation_loss.append(self.val_epoch_loss/max(1,self.m))
        self.reconstruction_training_loss.append(self.reconstruction_epoch_loss/max(1,self.n))
        self.reconstruction_validation_loss.append(self.val_reconstruction_epoch_loss/max(1,self.m))
        self.divergence_training_loss.append(self.divergence_epoch_loss/max(1,self.n))
        self.divergence_validation_loss.append(self.val_divergence_epoch_loss/max(1,self.m))
        self.burned_training_loss.append(self.burned_epoch_loss/max(1,self.n))
        self.burned_validation_loss.append(self.val_burned_epoch_loss/max(1,self.m))
        tr_s = (self.spearman_epoch_sum / max(1, self.spearman_epoch_cnt))
        va_s = (self.val_spearman_epoch_sum / max(1, self.val_spearman_epoch_cnt))
        tr_g = (self.gradnorm_epoch_sum   / max(1, self.gradnorm_epoch_cnt))
        va_g = (self.val_gradnorm_epoch_sum / max(1, self.val_gradnorm_epoch_cnt))
        self.spearman_training.append(tr_s); self.spearman_validation.append(va_s)
        self.gradnorm_training.append(tr_g);  self.gradnorm_validation.append(va_g)

        # reset
        self.epoch_loss = self.val_epoch_loss = 0
        self.reconstruction_epoch_loss = self.val_reconstruction_epoch_loss = 0
        self.divergence_epoch_loss = self.val_divergence_epoch_loss = 0
        self.burned_epoch_loss = self.val_burned_epoch_loss = 0
        self.m = self.n = 0
        self.spearman_epoch_sum = self.val_spearman_epoch_sum = 0.0
        self.spearman_epoch_cnt = self.val_spearman_epoch_cnt = 0
        self.gradnorm_epoch_sum = self.val_gradnorm_epoch_sum = 0.0
        self.gradnorm_epoch_cnt = self.val_gradnorm_epoch_cnt = 0


    def plot_loss(self, epochs):
        self.to("cpu")
        tag = self._params_tag()
        # (keep your plots) + alignment plots
        plt.ion(); fig = plt.figure()
        plt.plot(self.training_loss[1:], label='training loss')
        plt.plot(self.validation_loss[1:], label='validation loss')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        plt.title(f"Total Loss (instance={self.instance})")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"loss_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}__{tag}.png"
        )

        plt.figure()
        plt.plot(self.reconstruction_training_loss[1:], label='recon train')
        plt.plot(self.reconstruction_validation_loss[1:], label='recon val')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        plt.title(f"Reconstruction Loss")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"recon_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}__{tag}.png"
        )

        plt.figure()
        plt.plot(self.divergence_training_loss[1:], label='KL train')
        plt.plot(self.divergence_validation_loss[1:], label='KL val')
        plt.xlabel('Epochs'); plt.ylabel('KL'); plt.legend()
        plt.title(f"KL Divergence")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"kl_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}__{tag}.png"
        )

        plt.figure()
        plt.plot(self.burned_training_loss[1:], label='pred train')
        plt.plot(self.burned_validation_loss[1:], label='pred val')
        plt.xlabel('Epochs'); plt.ylabel('Pred loss'); plt.legend()
        plt.title(f"Predictor Loss")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"pred_sub20x20_latent={self.latent_dims}_capacity={self.c}_{epochs}__{tag}.png"
        )

        plt.figure()
        plt.plot(self.spearman_training[1:], label='ρ train')
        plt.plot(self.spearman_validation[1:], label='ρ val')
        plt.xlabel('Epochs'); plt.ylabel('ρ'); plt.legend()
        plt.title(f"Predictor vs True: Spearman")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"spearman_{epochs}__{tag}.png"
        )

        plt.figure()
        plt.plot(self.gradnorm_training[1:], label='‖∇z_free s‖ train')
        plt.plot(self.gradnorm_validation[1:], label='‖∇z_free s‖ val')
        plt.xlabel('Epochs'); plt.ylabel('Grad norm'); plt.legend()
        plt.title(f"Predictor Gradient Norm (FREE dims)")
        plt.tight_layout()
        plt.savefig(
            f"experiments/{self.instance}/train_stats/{self.name}/"
            f"gradnorm_{epochs}__{tag}.png"
        )

        # ===== NEW: structure metrics plots (per-epoch series) =====
        # 1) predictor vs true Spearman
        if len(self.struct_pred_spearman) > 1:
            plt.figure()
            plt.plot(self.struct_pred_spearman[1:], label='Spearman(pred,true)')
            plt.xlabel('Epochs'); plt.ylabel('ρ'); plt.legend()
            plt.title(f"Structure: Predictor vs True (Spearman)")
            plt.tight_layout()
            plt.savefig(
                f"experiments/{self.instance}/train_stats/{self.name}/"
                f"struct_pred_spearman_{epochs}__{tag}.png"
            )

        # 2) ridge Spearman and R^2
        if len(self.struct_ridge_spearman) > 1:
            plt.figure()
            plt.plot(self.struct_ridge_spearman[1:], label='Ridge ρ(z→f)')
            plt.plot(self.struct_ridge_R2[1:], label='Ridge R^2')
            plt.xlabel('Epochs'); plt.ylabel('score'); plt.legend()
            plt.title(f"Structure: Ridge Alignment (ρ & R²)")
            plt.tight_layout()
            plt.savefig(
                f"experiments/{self.instance}/train_stats/{self.name}/"
                f"struct_ridge_{epochs}__{tag}.png"
            )

        # 3) local Lipschitz
        if len(self.struct_lipschitz_med) > 1:
            plt.figure()
            plt.plot(self.struct_lipschitz_med[1:], label='Lipschitz median')
            plt.plot(self.struct_lipschitz_p90[1:], label='Lipschitz p90')
            plt.xlabel('Epochs'); plt.ylabel('Δf/‖Δz‖'); plt.legend()
            plt.title(f"Structure: Local Lipschitz")
            plt.tight_layout()
            plt.savefig(
                f"experiments/{self.instance}/train_stats/{self.name}/"
                f"struct_lipschitz_{epochs}__{tag}.png"
            )

        # 4) mutation power and best-of-32 gain proxy
        if len(self.struct_aSigmaa_free) > 1:
            plt.figure()
            plt.plot(self.struct_aSigmaa_free[1:], label='a^T Σ a (FREE)')
            plt.plot(self.struct_best_gain32[1:], label='best-of-32 Δf proxy')
            plt.xlabel('Epochs'); plt.ylabel('power / proxy'); plt.legend()
            plt.title(f"Structure: Mutation Power & Best-of-32 Proxy")
            plt.tight_layout()
            plt.savefig(
                f"experiments/{self.instance}/train_stats/{self.name}/"
                f"struct_mutation_power_{epochs}__{tag}.png"
            )

        # 5) topo & Moran’s I
        if len(self.struct_topo) > 1:
            plt.figure()
            plt.plot(self.struct_topo[1:], label='Topo ρ(dist, |Δf|)')
            plt.plot(self.struct_moransI[1:], label="Moran's I (kNN)")
            plt.xlabel('Epochs'); plt.ylabel('score'); plt.legend()
            plt.title(f"Structure: Topographic Similarity & Moran’s I")
            plt.tight_layout()
            plt.savefig(
                f"experiments/{self.instance}/train_stats/{self.name}/"
                f"struct_topo_moran_{epochs}__{tag}.png"
            )


    def calc_test_loss(self, output, images, r):
        return self.loss(output, images, r)
    
    

    def _fmt_val(self, v):
        if isinstance(v, float):
            return f"{v:.6g}"           # compact float
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v)

    def _sanitize_token(self, s: str) -> str:
        # keep only [A-Za-z0-9._-], replace others with '-'
        return re.sub(r"[^A-Za-z0-9._-]", "-", s)

    def _params_tag(self, max_len: int = 180) -> str:
        """
        k1=v1__k2=v2__... (sorted by key), sanitized for filenames.
        If too long, truncate and append short hash.
        """
        # Ensure we actually have the full dict
        params = ["latent_dims", "distribution_std", "variational_beta", "alpha"]
        items = []
        for k in sorted(self.params.keys()):
            if k in params:
                kv = f"{k}={str(self.params[k])}"
                items.append(kv)
        tag = "__".join(items) if items else "no_params"
        return tag

    @staticmethod
    def _to_numpy(x): 
        return x.detach().cpu().numpy()

    @staticmethod
    def _logit(p, eps=1e-6):
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    @torch.no_grad()
    def _encode_all(self, dataset, max_n=5000, batch_size=256, device=None, expected_C=4):
        device = device or self.device
        # Encode up to max_n items for speed
        N = min(max_n, len(dataset))
        idx = np.arange(len(dataset))
        if N < len(dataset):
            idx = np.random.default_rng(0).choice(len(dataset), size=N, replace=False)
        loader = torch.utils.data.DataLoader([dataset[i] for i in idx], batch_size=batch_size, shuffle=False)
        mu_all, sig_all, r_all = [], [], []
        for x, r in loader:
            # x: [B, 4, 20, 20]; r: [B] or [B,1]
            x = x.to(device)
            mu, log_sigma = self.encode(x)
            sigma = torch.exp(log_sigma)
            mu_all.append(mu)
            sig_all.append(sigma)
            r_all.append(r.view(-1).to(device))
        mu_all = torch.cat(mu_all, 0); sig_all = torch.cat(sig_all, 0); r_all = torch.cat(r_all, 0)
        return mu_all, sig_all, r_all

    @staticmethod
    def _nearest_neighbors(Z: np.ndarray, k: int):
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k+1, len(Z))).fit(Z)
            D, I = nn.kneighbors(Z, return_distance=True)
            return D[:,1:], I[:,1:]
        except Exception:
            Zt = torch.tensor(Z, dtype=torch.float32)
            d = torch.cdist(Zt, Zt, p=2)
            d.fill_diagonal_(float("inf"))
            D, I = torch.topk(d, k=min(k, Zt.shape[0]-1), largest=False, dim=1)
            return D.cpu().numpy(), I.cpu().numpy()

    @staticmethod
    def _ridge_metrics(Z: np.ndarray, y: np.ndarray, lam: float = 1e-2):
        Zm, Zs = Z.mean(0), Z.std(0) + 1e-8
        Zstd = (Z - Zm) / Zs
        ym, ys = y.mean(), y.std() + 1e-8
        ystd = (y - ym) / ys
        A = Zstd.T @ Zstd + lam * np.eye(Zstd.shape[1])
        w = np.linalg.solve(A, Zstd.T @ ystd)
        yhat = Zstd @ w
        rho, _ = spearmanr(yhat, y)
        r2 = 1.0 - np.sum((ystd - yhat) ** 2) / np.sum((ystd - ystd.mean()) ** 2)
        a = w / (np.linalg.norm(w) + 1e-12)  # unit direction
        covZ = np.cov(Z.T)
        dir_power = float(np.var(Z @ a) / (np.trace(covZ) + 1e-12))
        return dict(spearman=float(rho), r2=float(r2), w=w, a=a, dir_power=dir_power)

    def _local_lipschitz(self, Z: np.ndarray, y: np.ndarray, k: int = 20, q: float = 0.9):
        D, I = self._nearest_neighbors(Z, k)
        diffs = np.abs(y[:, None] - y[I])
        lips  = diffs / (D + 1e-12)
        lip_i = np.max(lips, axis=1)
        return dict(median=float(np.median(lip_i)), q90=float(np.quantile(lip_i, q)))

    @staticmethod
    def _topo_similarity(Z: np.ndarray, y: np.ndarray, n_pairs: int = 50000, seed: int = 0):
        rng = np.random.default_rng(seed)
        N = len(y)
        i = rng.integers(0, N, size=n_pairs)
        j = rng.integers(0, N, size=n_pairs)
        m = i != j
        i, j = i[m], j[m]
        dz = np.linalg.norm(Z[i] - Z[j], axis=1)
        dy = np.abs(y[i] - y[j])
        rho, _ = spearmanr(dz, dy)
        return float(rho)

    def _morans_I_knn(self, Z: np.ndarray, y: np.ndarray, k: int = 20):
        try:
            _, I = self._nearest_neighbors(Z, k)
            y0 = y - y.mean(); N = len(y)
            num = 0.0
            for idx, neigh in enumerate(I):
                num += y0[idx] * y0[neigh].sum()
            W = I.size  # directed edges
            den = float((y0 @ y0) + 1e-12)
            return float((N / (W + 1e-12)) * (num / den))
        except Exception:
            return float("nan")

    @staticmethod
    def _mutation_power(a_unit: np.ndarray, sigmas: np.ndarray, free_start: int):
        s2 = (sigmas ** 2).mean(axis=0)  # [D]
        a_f, s2_f = a_unit[free_start:], s2[free_start:]
        return float((a_f * a_f * s2_f).sum())

    @staticmethod
    def _expected_best_of_M_gain(a_sigma_a: float, M: int = 32):
        return float(np.sqrt(max(1e-12, 2.0 * np.log(max(2, M)))) * np.sqrt(max(0.0, a_sigma_a)))

    @torch.no_grad()
    def update_structure_metrics(self, dataset, max_n=3000, batch_size=256, M=32):
        """Compute structure metrics on a subset of `dataset` and append to time series."""
        self.eval()
        mu_all, sig_all, r_true = self._encode_all(dataset, max_n=max_n, batch_size=batch_size, device=self.device)
        Z = self._to_numpy(mu_all)
        Sig = self._to_numpy(sig_all)
        y = self._to_numpy(r_true.view(-1))

        # predictor scores (raw if rank-loss mode; else logit(prob))
        raw = self._predict_burned_raw(mu_all)
        s_pred = raw if self.use_rank_loss else self._logit(torch.sigmoid(raw))
        s = self._to_numpy(s_pred)

        rho_pred, _ = spearmanr(s, y)
        rid = self._ridge_metrics(Z, y, lam=1e-2)
        lip = self._local_lipschitz(Z, y, k=20, q=0.9)
        topo = self._topo_similarity(Z, y, n_pairs=50000, seed=0)
        morI = self._morans_I_knn(Z, y, k=20)
        free_start = getattr(self, "rec_dim", 0)
        aSigmaa = self._mutation_power(rid["a"], Sig, free_start=free_start)
        best_gain = self._expected_best_of_M_gain(aSigmaa, M=M)

        # append to series
        self.struct_pred_spearman.append(float(rho_pred))
        self.struct_ridge_spearman.append(float(rid["spearman"]))
        self.struct_ridge_R2.append(float(rid["r2"]))
        self.struct_lipschitz_med.append(float(lip["median"]))
        self.struct_lipschitz_p90.append(float(lip["q90"]))
        self.struct_topo.append(float(topo))
        self.struct_moransI.append(float(morI))
        self.struct_aSigmaa_free.append(float(aSigmaa))
        self.struct_best_gain32.append(float(best_gain))
