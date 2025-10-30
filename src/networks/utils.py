import os, numpy as np
import torch
from scipy.stats import spearmanr
from numpy import genfromtxt
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from src.algorithms.GA.utils import write_firewall_file, erase_firebreaks
import os, json, numpy as np
import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, MeanAbsolutePercentageError


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ====== imports you likely already have ======


# ====== small utils ======
def _ensure_b1hw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x is [B,1,20,20]; accepts [B,20,20] or [1,20,20] or [B,1,20,20].
    """
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    return x


# ====== encode a subset (aggregate posterior stats) ======
@torch.no_grad()
def encode_dataset(model, dataset, max_n=2048, batch_size=256, device="cuda",
                   expected_C: int = 4, multi2one: str = "mean", one2multi: str = "tile"):
    """
    Returns: mu_all [N,D], sigma_all [N,D] with N<=max_n.
    """
    model.eval()
    device = model.device
    N = min(max_n, len(dataset))
    idxs = np.random.default_rng(123).choice(len(dataset), size=N, replace=False)
    mus, sigs = [], []
    for s in range(0, N, batch_size):
        batch_idx = idxs[s:s+batch_size]
        xs = []
        for i in batch_idx:
            x_i = dataset[i][0]                   # (C,H,W)
            xs.append(x_i)
        x = torch.stack(xs, dim=0).to(device)       # [B, expected_C, 20, 20]
        mu, log_sigma = model.encode(x)             # [B,D], [B,D] (log σ)
        sigma = torch.exp(log_sigma)
        mus.append(mu.detach()); sigs.append(sigma.detach())
    mu_all = torch.cat(mus, dim=0)
    sigma_all = torch.cat(sigs, dim=0)
    return mu_all, sigma_all

# ====== sampling from prior / aggregate posterior / posterior mean ======
@torch.no_grad()
def sample_z(mu_all: torch.Tensor, sigma_all: torch.Tensor, n: int,
             mode: str = "prior", device: str = None) -> torch.Tensor:
    """
    mode in {"prior", "agg", "posterior_mean", "posterior_sample"}.
    Returns z [n,D].
    """
    device = device or mu_all.device
    N, D = mu_all.shape
    if mode == "prior":
        return torch.randn(n, D, device=device)
    # pick component indices for mixture modes
    idx = torch.randint(low=0, high=N, size=(n,), device=device)
    mu = mu_all[idx]               # [n,D]
    sig = sigma_all[idx]           # [n,D]
    if mode == "posterior_mean":
        return mu
    if mode in ("agg", "posterior_sample"):
        eps = torch.randn_like(mu)
        return mu + eps * sig
    raise ValueError(f"Unknown mode '{mode}'")

# ====== predictor wrapper ======
@torch.no_grad()
def predict_score_ccvae(ccvae_model, z: torch.Tensor, use_raw: bool = True) -> torch.Tensor:
    """
    If use_raw=True and model exposes _predict_burned_raw, use it (better numeric range for rank).
    Else fall back to predict_burned (prob).
    Returns [B] tensor.
    """
    ccvae_model.eval()
    if use_raw and hasattr(ccvae_model, "_predict_burned_raw"):
        out = ccvae_model._predict_burned_raw(z)    # [B]
    else:
        out = ccvae_model.predict_burned(z).view(-1)
    return out.detach()

# ====== decode -> fitness (vectorized over batches) ======
@torch.no_grad()
def evaluate_true_fitness(model_with_decoder, z: torch.Tensor, fitness_fn, batch_size=128) -> torch.Tensor:
    """
    Returns [B] fitness for latent batch z.
    """
    model_with_decoder.eval()
    vals = []
    for s in range(0, z.shape[0], batch_size):
        zb = z[s:s+batch_size]
        xhat = model_with_decoder.decode(zb)        # [B,1,20,20]
        f = fitness_fn(xhat)                        # [B] (your function returns per-sample)
        vals.append(f.detach().view(-1))
    return torch.cat(vals, dim=0)

# ====== correlation from different z sources ======
@torch.no_grad()
def correlation_check(ccvae_model, latent_provider_model, dataset, fitness_fn,
                      n_agg=10, n_z=10, device="cuda",
                      modes=("prior","agg","posterior_mean","posterior_sample"),
                      plot=True):
    """
    For each mode, sample z, compute ŝ(z) vs f(g(z)), and Spearman ρ.
    Prints and returns a dict of results.
    """
    mu_all, sig_all = encode_dataset(latent_provider_model, dataset, max_n=n_agg, device=device,
                                     expected_C=4, multi2one="mean", one2multi="tile")
    results = {}
    for mode in modes:
        z = sample_z(mu_all, sig_all, n=n_z, mode=mode, device=device)
        s_pred = predict_score_ccvae(ccvae_model, z, use_raw=True).cpu().numpy()
        f_true = evaluate_true_fitness(latent_provider_model, z, fitness_fn).cpu().numpy()
        rho, p = spearmanr(s_pred, f_true)
        if plot:
            plt.figure(figsize=(5.2,4.2))
            plt.scatter(s_pred, f_true, s=10, alpha=0.4)
            plt.xlabel("ŝ(z)"); plt.ylabel("f(g(z))"); plt.title(f"{mode} ρ={rho:.3f}")
            plt.tight_layout(); plt.show()
            plt.savefig(f"experiments/homo_2/train_stats/{ccvae_model.name}/correlation_score_{mode}_{ccvae_model._params_tag()}.png")
        results[mode] = dict(rho=float(rho), p=float(p), n=int(n_z))
    return results

# ====== on-manifold correlation (μ(x)) ======
@torch.no_grad()
def correlation_on_manifold(ccvae_model, dataset, max_n=4000, batch_size=256, device="cuda", plot=False):
    """
    ρ between ŝ(μ(x)) and f(g(μ(x))) over a dataset subset (validation).
    """
    ccvae_model.eval()
    N = min(max_n, len(dataset))
    idx = np.random.default_rng(0).choice(len(dataset), size=N, replace=False)
    s_all, f_all = [], []
    for s in range(0, N, batch_size):
        ii = idx[s:s+batch_size]
        x = torch.stack([dataset[i][0] for i in ii], dim=0).to(device) # [B,4,20,20]
        mu, _ = ccvae_model.encode(x)
        xhat = ccvae_model.decode(mu)
        f = fitness_fn(xhat)                             # [B]
        sp = predict_score_ccvae(ccvae_model, mu, use_raw=True)
        s_all.append(sp.cpu()); f_all.append(f.cpu())
    s_all = torch.cat(s_all).numpy(); f_all = torch.cat(f_all).numpy()
    rho, p = spearmanr(s_all, f_all)
    print(f"[On-manifold] Spearman(pred,true) = {rho:.3f} (p={p:.2g})")
    if plot:
        plt.figure(figsize=(5.2,4.2))
        plt.scatter(s_all, f_all, s=10, alpha=0.4)
        plt.xlabel("ŝ(μ(x))"); plt.ylabel("f(g(μ(x)))"); plt.title(f"ρ={rho:.3f}")
        plt.tight_layout(); plt.show()
        plt.savefig(f"experiments/homo_2/train_stats/{ccvae_model.name}/correlation_score_manifold_{ccvae_model._params_tag()}.png")
    return dict(rho=float(rho), p=float(p), n=int(N))


@torch.no_grad()
def reconstruction_metrics(ccvae_model,
                                 dataset,
                                 max_samples: int = 45000,
                                 topk: int = 20,
                                 threshold: float = 0.5,
                                 save_path: str = None):
    """
    For each (x, r) in `dataset`:
      1) ŷ_map = model(x)[0]  -> build a binary top-K mask from ŷ_map
      2) Update binary map metrics (Accuracy/Precision/Recall/F1) vs target mask x[:,0]
      3) Update regression metrics (MSE/MAE/R2/MAPE) on predicted scalar r̂ vs r
      4) Accumulate BCE(recon, target) and top-K overlap/Jaccard

    Saves a JSON with keys exactly as requested.
    """
    ccvae_model.eval()
    dev = next(ccvae_model.parameters()).device
    ccvae_model.to()
    # ---- metrics on correct device ----
    acc  = Accuracy(task="binary", threshold=threshold).to(dev)
    prec = Precision(task="binary", threshold=threshold).to(dev)
    rec  = Recall(task="binary", threshold=threshold).to(dev)
    f1   = F1Score(task="binary", threshold=threshold).to(dev)

    mse  = MeanSquaredError().to(dev)
    mae  = MeanAbsoluteError().to(dev)
    r2   = R2Score().to(dev)
    mape = MeanAbsolutePercentageError().to(dev)

    N = min(max_samples, len(dataset))

    n = 0
    for i in range(len(dataset)):
        x, r = dataset[i]              # x: [C,20,20], r: scalar or [1]
        xb = x.unsqueeze(0).to(dev)    # [1,C,20,20]
        rb = r.view(1).to(dev)         # [1]

        recon, r_pred = ccvae_model(xb, rb)   # recon: [1,1,20,20]; r_pred: [1] or [1,1]
        r_pred = r_pred.view(1)

        # target mask = first channel of x
        target = xb[:, :1, :, :]                 # [1,1,20,20]
        target_mask = target.squeeze(1)          # [1,20,20]; assume binary (or threshold inside metrics)



        # build binary mask of top-k cells from recon
        flat = recon.view(-1)                    # [400]
        idx  = torch.topk(flat, topk, dim=0).indices
        rows = idx // 20
        cols = idx % 20
        pred_mask = torch.zeros((1, 20, 20), device=dev, dtype=recon.dtype)
        pred_mask[0, rows, cols] = 1.0

        # ---- binary map metrics ----
        acc.update(pred_mask, target_mask)
        prec.update(pred_mask, target_mask)
        rec.update(pred_mask, target_mask)
        f1.update(pred_mask, target_mask)

        # ---- regression metrics on scalar score ----
        mse.update(r_pred, rb)
        mae.update(r_pred, rb)
        r2.update(r_pred, rb)
        # clamp to avoid div by 0 in MAPE
        mape.update(r_pred, torch.clamp(rb, min=1e-8))

        n += 1
        if n >= N:
            break

    # ---- finalize numbers ----
    out = {
        "accuracy":      float(acc.compute().cpu()),
        "precision":     float(prec.compute().cpu()),
        "recall":        float(rec.compute().cpu()),
        "f1":            float(f1.compute().cpu()),
        "mse":           float(mse.compute().cpu()),
        "mae":           float(mae.compute().cpu()),
        "r2":            float(r2.compute().cpu()),
        "mape":          float(mape.compute().cpu()),
    }

    # ---- save JSON exactly as requested ----
    base = f"experiments/homo_2/train_stats/{ccvae_model.name}/"
    tag = ccvae_model._params_tag() if hasattr(ccvae_model, "_params_tag") else "no_params"
    save_path = os.path.join(base, f"rec_metrics_{tag}.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"metrics": out}, f, indent=2)
    return out, save_path


# ====== your fitness function (batched) ======
def fitness_fn(x_hat: torch.Tensor, n_sims: int = 50) -> torch.Tensor:
    """
    Returns fitness per sample as a [B]-tensor.
    For each sample: select top-20 cells from [1,20,20] map and run Cell2Fire.
    NOTE: this calls external binaries and disk IO; keep batch sizes modest.
    """
    import os
    from numpy import genfromtxt

    x_hat = _ensure_b1hw(x_hat).detach().cpu()  # [B,1,20,20]
    B = x_hat.shape[0]

    weathers_dir = "../data/complete_random/homo_2/Sub20x20/Weathers/"
    n_weathers = len([f for f in os.listdir(weathers_dir) if f.endswith(".csv")]) - 2

    exe = "../src/algorithms/eval/C2F-W/Cell2Fire/Cell2Fire"
    input_folder = "../data/complete_random/homo_2/Sub20x20/"
    output_folder = "../src/algorithms/eval/results/"
    base_grids_dir = os.path.join(output_folder, "Grids", "Grids")
    firebreak_path = "../src/algorithms/eval/harvested/HarvestedCells.csv"

    rewards = []
    for b in range(B):
        grid = x_hat[b, 0]  # [20,20]
        flat = grid.flatten()
        _, idx = torch.topk(flat, 20)
        rr, cc = np.unravel_index(idx.numpy(), (20, 20))
        matrix = torch.zeros((20, 20), dtype=torch.float32)
        matrix[rr, cc] = 1.0
        assert matrix.sum().item() == 20

        # user-provided helpers must exist:
        write_firewall_file(matrix * -1.0)

        cmd = (
            f"{exe} --input-instance-folder {input_folder} --output-folder {output_folder} "
            f"--sim-years 1 --nsims {n_sims} --Fire-Period-Length 1.0 --output-messages "
            f"--ROS-CV 2.0 --seed 123 --weather random --ignitions --IgnitionRad 4 "
            f"--sim C --final-grid --nweathers {n_weathers} --FirebreakCells {firebreak_path}"
        )
        os.system(cmd + " >/dev/null 2>&1")

        reward = 0
        for j in range(1, n_sims + 1):
            dir_j = f"{base_grids_dir}{j}/"
            if not os.path.isdir(dir_j):
                continue
            files = sorted(os.listdir(dir_j))
            if not files:
                continue
            last_file = files[-1]
            my_data = genfromtxt(os.path.join(dir_j, last_file), delimiter=',')
            reward -= int((my_data == 1).sum())
        erase_firebreaks()

        rewards.append(1.0 + ((reward / n_sims) / 400.0))

    return torch.tensor(rewards, dtype=torch.float32)
