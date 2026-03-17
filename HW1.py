# ============================================================
# PINN Assignment Full Solution Code
# Problem:
#    du/dt = -a u,   u(0)=1,   t in [0,1]
# Includes:
#   1) Regression (MLP)
#   2) Forward PINN
#   3) Inverse PINN
#   4) K-fold CV for Regression / Inverse PINN (manual implementation)
#   5) Requested analysis plots
# ============================================================

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. Reproducibility
# ============================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)


# ============================================================
# 1. Problem setup
# ============================================================
def true_solution(t, a):
    return np.exp(-a * t)

def true_solution_torch(t, a):
    return torch.exp(-a * t)

def relative_l2_error_torch(pred, true, eps=1e-12):
    num = torch.sqrt(torch.sum((pred - true) ** 2))
    den = torch.sqrt(torch.sum(true ** 2)) + eps
    return (num / den).item()

def relative_l2_error_numpy(pred, true, eps=1e-12):
    num = np.sqrt(np.sum((pred - true) ** 2))
    den = np.sqrt(np.sum(true ** 2)) + eps
    return num / den


def generate_dataset(
    n_points=50,
    a=2.0,
    t_min=0.0,
    t_max=1.0,
    noise_std=0.0,
    sort=True
):
    t = np.random.uniform(t_min, t_max, size=(n_points, 1))
    if sort:
        t = np.sort(t, axis=0)

    u_true = true_solution(t, a)
    noise = np.random.normal(0.0, noise_std, size=u_true.shape)
    u_obs = u_true + noise

    t_tensor = torch.tensor(t, dtype=torch.float32)
    u_true_tensor = torch.tensor(u_true, dtype=torch.float32)
    u_obs_tensor = torch.tensor(u_obs, dtype=torch.float32)
    return t_tensor, u_true_tensor, u_obs_tensor


def make_uniform_grid(n_points=200, t_min=0.0, t_max=1.0):
    t = np.linspace(t_min, t_max, n_points).reshape(-1, 1)
    return torch.tensor(t, dtype=torch.float32)


# ============================================================
# 2. Shared MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, depth=2, out_dim=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Loss functions
# ============================================================
def regression_loss(model, t_data, u_data):
    u_pred = model(t_data)
    loss = F.mse_loss(u_pred, u_data)
    return loss


def compute_du_dt(model, t):
    """
    Compute du/dt using autograd.
    t must require grad.
    """
    u = model(t)
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    return u, du_dt


def forward_pinn_loss(model, t_data, u_data, t_collocation, a, w_data=1.0, w_phys=1.0):
    # data loss
    u_pred_data = model(t_data)
    loss_data = F.mse_loss(u_pred_data, u_data)

    # physics loss
    t_col = t_collocation.clone().detach().requires_grad_(True)
    u_col, du_dt = compute_du_dt(model, t_col)
    residual = du_dt + a * u_col
    loss_phys = torch.mean(residual ** 2)

    # initial condition
    t0 = torch.zeros((1, 1), dtype=torch.float32, device=t_data.device)
    u0_pred = model(t0)
    loss_ic = F.mse_loss(u0_pred, torch.ones_like(u0_pred))

    loss = w_data * loss_data + w_phys * loss_phys + loss_ic
    info = {
        "loss_total": loss.item(),
        "loss_data": loss_data.item(),
        "loss_phys": loss_phys.item(),
        "loss_ic": loss_ic.item(),
    }
    return loss, info


def inverse_pinn_loss(model, t_data, u_data, t_collocation, raw_a, w_data=1.0, w_phys=1.0):
    # positive parameterization
    a_hat = F.softplus(raw_a)

    # data loss
    u_pred_data = model(t_data)
    loss_data = F.mse_loss(u_pred_data, u_data)

    # physics loss
    t_col = t_collocation.clone().detach().requires_grad_(True)
    u_col, du_dt = compute_du_dt(model, t_col)
    residual = du_dt + a_hat * u_col
    loss_phys = torch.mean(residual ** 2)

    # initial condition
    t0 = torch.zeros((1, 1), dtype=torch.float32, device=t_data.device)
    u0_pred = model(t0)
    loss_ic = F.mse_loss(u0_pred, torch.ones_like(u0_pred))

    loss = w_data * loss_data + w_phys * loss_phys + loss_ic
    info = {
        "loss_total": loss.item(),
        "loss_data": loss_data.item(),
        "loss_phys": loss_phys.item(),
        "loss_ic": loss_ic.item(),
        "a_hat": a_hat.item(),
    }
    return loss, info


# ============================================================
# 4. Training utilities
# ============================================================
def train_regression(
    model,
    t_train,
    u_train,
    epochs=2000,
    lr=1e-3,
    verbose=False
):
    model = model.to(device)
    t_train = t_train.to(device)
    u_train = u_train.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = regression_loss(model, t_train, u_train)
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())

        if verbose and ((epoch + 1) % 500 == 0):
            print(f"[Regression] epoch={epoch+1:5d}, loss={loss.item():.6e}")

    return model, history


def train_forward_pinn(
    model,
    t_data,
    u_data,
    t_collocation,
    a,
    epochs=3000,
    lr=1e-3,
    w_data=1.0,
    w_phys=1.0,
    verbose=False
):
    model = model.to(device)
    t_data = t_data.to(device)
    u_data = u_data.to(device)
    t_collocation = t_collocation.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss_total": [], "loss_data": [], "loss_phys": [], "loss_ic": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss, info = forward_pinn_loss(
            model, t_data, u_data, t_collocation, a, w_data=w_data, w_phys=w_phys
        )
        loss.backward()
        optimizer.step()

        for k in history.keys():
            history[k].append(info[k])

        if verbose and ((epoch + 1) % 500 == 0):
            print(
                f"[Forward PINN] epoch={epoch+1:5d}, "
                f"total={info['loss_total']:.6e}, "
                f"data={info['loss_data']:.6e}, "
                f"phys={info['loss_phys']:.6e}, "
                f"ic={info['loss_ic']:.6e}"
            )

    return model, history


def train_inverse_pinn(
    model,
    t_data,
    u_data,
    t_collocation,
    raw_a_init=0.5,
    epochs=3000,
    lr_model=1e-3,
    lr_param=1e-3,
    w_data=1.0,
    w_phys=1.0,
    verbose=False
):
    model = model.to(device)
    t_data = t_data.to(device)
    u_data = u_data.to(device)
    t_collocation = t_collocation.to(device)

    raw_a = nn.Parameter(torch.tensor([raw_a_init], dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": lr_model},
            {"params": [raw_a], "lr": lr_param},
        ]
    )

    history = {
        "loss_total": [],
        "loss_data": [],
        "loss_phys": [],
        "loss_ic": [],
        "a_hat": [],
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss, info = inverse_pinn_loss(
            model, t_data, u_data, t_collocation, raw_a,
            w_data=w_data, w_phys=w_phys
        )
        loss.backward()
        optimizer.step()

        for k in history.keys():
            history[k].append(info[k])

        if verbose and ((epoch + 1) % 500 == 0):
            print(
                f"[Inverse PINN] epoch={epoch+1:5d}, "
                f"total={info['loss_total']:.6e}, "
                f"data={info['loss_data']:.6e}, "
                f"phys={info['loss_phys']:.6e}, "
                f"ic={info['loss_ic']:.6e}, "
                f"a_hat={info['a_hat']:.6f}"
            )

    a_hat = F.softplus(raw_a).item()
    return model, a_hat, history


# ============================================================
# 5. Evaluation utilities
# ============================================================
@torch.no_grad()
def evaluate_solution_error(model, a_true, n_eval=200):
    model.eval()
    t_eval = make_uniform_grid(n_eval).to(device)
    u_pred = model(t_eval)
    u_true = true_solution_torch(t_eval, a_true)
    err = relative_l2_error_torch(u_pred, u_true)
    return err, t_eval.cpu().numpy(), u_true.cpu().numpy(), u_pred.cpu().numpy()


def parameter_relative_error(a_true, a_pred, eps=1e-12):
    return abs(a_true - a_pred) / (abs(a_true) + eps)


# ============================================================
# 6. Manual K-fold split
# ============================================================
def build_kfold_indices(n_samples, k=5, shuffle=True, seed=0):
    """
    직접 K-fold index를 생성하는 함수
    반환:
        folds = [(train_idx, val_idx), ...]
    """
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start = current
        end = current + fold_size

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        folds.append((train_idx, val_idx))
        current = end

    return folds


# ============================================================
# 7. K-fold Cross Validation
# ============================================================
def kfold_regression(
    t_all,
    u_all,
    k=5,
    hidden_dim=32,
    depth=2,
    epochs=2000,
    lr=1e-3,
    seed=0
):
    t_np = t_all.numpy()
    u_np = u_all.numpy()

    folds = build_kfold_indices(len(t_np), k=k, shuffle=True, seed=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        t_train = torch.tensor(t_np[train_idx], dtype=torch.float32)
        u_train = torch.tensor(u_np[train_idx], dtype=torch.float32)
        t_val = torch.tensor(t_np[val_idx], dtype=torch.float32)
        u_val = torch.tensor(u_np[val_idx], dtype=torch.float32)

        model = MLP(hidden_dim=hidden_dim, depth=depth)
        model, history = train_regression(
            model, t_train, u_train, epochs=epochs, lr=lr, verbose=False
        )

        model.eval()
        with torch.no_grad():
            pred_val = model(t_val.to(device)).cpu()
            val_mse = F.mse_loss(pred_val, u_val).item()

        fold_results.append({
            "fold": fold + 1,
            "val_mse": val_mse,
            "history": history
        })

    val_mses = [x["val_mse"] for x in fold_results]
    return {
        "fold_results": fold_results,
        "mean_val_mse": float(np.mean(val_mses)),
        "std_val_mse": float(np.std(val_mses))
    }


def kfold_inverse_pinn(
    t_all,
    u_all,
    a_true,
    n_collocation=100,
    k=5,
    hidden_dim=32,
    depth=2,
    epochs=3000,
    lr_model=1e-3,
    lr_param=1e-3,
    w_data=1.0,
    w_phys=1.0,
    seed=0
):
    t_np = t_all.numpy()
    u_np = u_all.numpy()

    folds = build_kfold_indices(len(t_np), k=k, shuffle=True, seed=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        t_train = torch.tensor(t_np[train_idx], dtype=torch.float32)
        u_train = torch.tensor(u_np[train_idx], dtype=torch.float32)
        t_val = torch.tensor(t_np[val_idx], dtype=torch.float32)
        u_val = torch.tensor(u_np[val_idx], dtype=torch.float32)

        t_col = make_uniform_grid(n_collocation)

        model = MLP(hidden_dim=hidden_dim, depth=depth)
        model, a_hat, history = train_inverse_pinn(
            model,
            t_train,
            u_train,
            t_col,
            raw_a_init=0.5,
            epochs=epochs,
            lr_model=lr_model,
            lr_param=lr_param,
            w_data=w_data,
            w_phys=w_phys,
            verbose=False
        )

        model.eval()
        with torch.no_grad():
            pred_val = model(t_val.to(device)).cpu()
            val_mse = F.mse_loss(pred_val, u_val).item()

        sol_err, _, _, _ = evaluate_solution_error(model, a_true)
        a_err = parameter_relative_error(a_true, a_hat)

        fold_results.append({
            "fold": fold + 1,
            "val_mse": val_mse,
            "solution_rel_l2": sol_err,
            "param_rel_error": a_err,
            "a_hat": a_hat,
            "history": history
        })

    val_mses = [x["val_mse"] for x in fold_results]
    sol_errs = [x["solution_rel_l2"] for x in fold_results]
    param_errs = [x["param_rel_error"] for x in fold_results]

    return {
        "fold_results": fold_results,
        "mean_val_mse": float(np.mean(val_mses)),
        "std_val_mse": float(np.std(val_mses)),
        "mean_solution_rel_l2": float(np.mean(sol_errs)),
        "std_solution_rel_l2": float(np.std(sol_errs)),
        "mean_param_rel_error": float(np.mean(param_errs)),
        "std_param_rel_error": float(np.std(param_errs)),
    }


# ============================================================
# 8. Experiment 1: Regression
#    train data 개수에 따른 relative l2 error
# ============================================================
def experiment_regression_by_train_size(
    train_sizes,
    a_true=2.0,
    noise_std=0.0,
    hidden_dim=32,
    depth=2,
    epochs=2000,
    lr=1e-3,
    repeats=3
):
    results = []

    for n_train in train_sizes:
        errs = []
        for rep in range(repeats):
            set_seed(rep)
            t_train, u_true_train, u_obs_train = generate_dataset(
                n_points=n_train,
                a=a_true,
                noise_std=noise_std
            )

            model = MLP(hidden_dim=hidden_dim, depth=depth)
            model, _ = train_regression(
                model,
                t_train,
                u_obs_train,
                epochs=epochs,
                lr=lr,
                verbose=False
            )

            sol_err, _, _, _ = evaluate_solution_error(model, a_true)
            errs.append(sol_err)

        results.append({
            "n_train": n_train,
            "mean_solution_rel_l2": float(np.mean(errs)),
            "std_solution_rel_l2": float(np.std(errs)),
        })

    return results


def plot_regression_by_train_size(results):
    x = [r["n_train"] for r in results]
    y = [r["mean_solution_rel_l2"] for r in results]
    yerr = [r["std_solution_rel_l2"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=4)
    plt.xlabel("Number of training data")
    plt.ylabel("Relative L2 error of solution")
    plt.title("Regression: Train data size vs Relative L2 error")
    plt.grid(True)
    plt.show()


# ============================================================
# 9. Experiment 2: Forward PINN
#    collocation point 개수에 따른 relative l2 error
# ============================================================
def experiment_forward_pinn_by_collocation(
    n_ref=20,
    collocation_sizes=(10, 20, 50, 100, 200),
    a_true=2.0,
    noise_std=0.0,
    hidden_dim=32,
    depth=2,
    epochs=3000,
    lr=1e-3,
    w_data=1.0,
    w_phys=1.0,
    repeats=3
):
    results = []

    for n_col in collocation_sizes:
        errs = []
        for rep in range(repeats):
            set_seed(rep)
            t_ref, u_true_ref, u_obs_ref = generate_dataset(
                n_points=n_ref,
                a=a_true,
                noise_std=noise_std
            )
            t_col = make_uniform_grid(n_col)

            model = MLP(hidden_dim=hidden_dim, depth=depth)
            model, _ = train_forward_pinn(
                model,
                t_ref,
                u_obs_ref,
                t_col,
                a=a_true,
                epochs=epochs,
                lr=lr,
                w_data=w_data,
                w_phys=w_phys,
                verbose=False
            )

            sol_err, _, _, _ = evaluate_solution_error(model, a_true)
            errs.append(sol_err)

        results.append({
            "n_collocation": n_col,
            "mean_solution_rel_l2": float(np.mean(errs)),
            "std_solution_rel_l2": float(np.std(errs)),
        })

    return results


def plot_forward_pinn_by_collocation(results):
    x = [r["n_collocation"] for r in results]
    y = [r["mean_solution_rel_l2"] for r in results]
    yerr = [r["std_solution_rel_l2"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=4)
    plt.xlabel("Number of collocation points")
    plt.ylabel("Relative L2 error of solution")
    plt.title("Forward PINN: Collocation size vs Relative L2 error")
    plt.grid(True)
    plt.show()


# ============================================================
# 10. Experiment 3: Inverse PINN
#     (A) 고정 reference data, varying collocation
#     (B) 고정 collocation, varying reference data
# ============================================================
def experiment_inverse_pinn_fixed_ref_vary_col(
    n_ref=20,
    collocation_sizes=(10, 20, 50, 100, 200),
    a_true=2.0,
    noise_std=0.0,
    hidden_dim=32,
    depth=2,
    epochs=4000,
    lr_model=1e-3,
    lr_param=1e-3,
    w_data=1.0,
    w_phys=1.0,
    repeats=3
):
    results = []

    for n_col in collocation_sizes:
        sol_errs = []
        param_errs = []

        for rep in range(repeats):
            set_seed(rep)
            t_ref, u_true_ref, u_obs_ref = generate_dataset(
                n_points=n_ref,
                a=a_true,
                noise_std=noise_std
            )
            t_col = make_uniform_grid(n_col)

            model = MLP(hidden_dim=hidden_dim, depth=depth)
            model, a_hat, _ = train_inverse_pinn(
                model,
                t_ref,
                u_obs_ref,
                t_col,
                raw_a_init=0.5,
                epochs=epochs,
                lr_model=lr_model,
                lr_param=lr_param,
                w_data=w_data,
                w_phys=w_phys,
                verbose=False
            )

            sol_err, _, _, _ = evaluate_solution_error(model, a_true)
            p_err = parameter_relative_error(a_true, a_hat)

            sol_errs.append(sol_err)
            param_errs.append(p_err)

        results.append({
            "n_collocation": n_col,
            "mean_solution_rel_l2": float(np.mean(sol_errs)),
            "std_solution_rel_l2": float(np.std(sol_errs)),
            "mean_param_rel_error": float(np.mean(param_errs)),
            "std_param_rel_error": float(np.std(param_errs)),
        })

    return results


def plot_inverse_pinn_fixed_ref_vary_col(results):
    x = [r["n_collocation"] for r in results]

    y1 = [r["mean_solution_rel_l2"] for r in results]
    e1 = [r["std_solution_rel_l2"] for r in results]

    y2 = [r["mean_param_rel_error"] for r in results]
    e2 = [r["std_param_rel_error"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y1, yerr=e1, marker='o', capsize=4)
    plt.xlabel("Number of collocation points")
    plt.ylabel("Relative L2 error of solution")
    plt.title("Inverse PINN (fixed ref): Collocation size vs Solution error")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y2, yerr=e2, marker='o', capsize=4)
    plt.xlabel("Number of collocation points")
    plt.ylabel("Relative error of parameter a")
    plt.title("Inverse PINN (fixed ref): Collocation size vs Parameter error")
    plt.grid(True)
    plt.show()


def experiment_inverse_pinn_fixed_col_vary_ref(
    n_col=100,
    ref_sizes=(5, 10, 20, 50, 100),
    a_true=2.0,
    noise_std=0.0,
    hidden_dim=32,
    depth=2,
    epochs=4000,
    lr_model=1e-3,
    lr_param=1e-3,
    w_data=1.0,
    w_phys=1.0,
    repeats=3
):
    results = []

    for n_ref in ref_sizes:
        sol_errs = []
        param_errs = []

        for rep in range(repeats):
            set_seed(rep)
            t_ref, u_true_ref, u_obs_ref = generate_dataset(
                n_points=n_ref,
                a=a_true,
                noise_std=noise_std
            )
            t_col = make_uniform_grid(n_col)

            model = MLP(hidden_dim=hidden_dim, depth=depth)
            model, a_hat, _ = train_inverse_pinn(
                model,
                t_ref,
                u_obs_ref,
                t_col,
                raw_a_init=0.5,
                epochs=epochs,
                lr_model=lr_model,
                lr_param=lr_param,
                w_data=w_data,
                w_phys=w_phys,
                verbose=False
            )

            sol_err, _, _, _ = evaluate_solution_error(model, a_true)
            p_err = parameter_relative_error(a_true, a_hat)

            sol_errs.append(sol_err)
            param_errs.append(p_err)

        results.append({
            "n_ref": n_ref,
            "mean_solution_rel_l2": float(np.mean(sol_errs)),
            "std_solution_rel_l2": float(np.std(sol_errs)),
            "mean_param_rel_error": float(np.mean(param_errs)),
            "std_param_rel_error": float(np.std(param_errs)),
        })

    return results


def plot_inverse_pinn_fixed_col_vary_ref(results):
    x = [r["n_ref"] for r in results]

    y1 = [r["mean_solution_rel_l2"] for r in results]
    e1 = [r["std_solution_rel_l2"] for r in results]

    y2 = [r["mean_param_rel_error"] for r in results]
    e2 = [r["std_param_rel_error"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y1, yerr=e1, marker='o', capsize=4)
    plt.xlabel("Number of reference data")
    plt.ylabel("Relative L2 error of solution")
    plt.title("Inverse PINN (fixed collocation): Ref size vs Solution error")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, y2, yerr=e2, marker='o', capsize=4)
    plt.xlabel("Number of reference data")
    plt.ylabel("Relative error of parameter a")
    plt.title("Inverse PINN (fixed collocation): Ref size vs Parameter error")
    plt.grid(True)
    plt.show()


# ============================================================
# 11. Example run blocks
# ============================================================
def demo_regression(a_true=2.0, noise_std=0.0):
    print("\n====================")
    print("Demo: Regression")
    print("====================")

    t_train, u_true_train, u_obs_train = generate_dataset(
        n_points=50, a=a_true, noise_std=noise_std
    )

    model = MLP(hidden_dim=32, depth=2)
    model, history = train_regression(
        model, t_train, u_obs_train, epochs=2000, lr=1e-3, verbose=True
    )

    sol_err, t_eval, u_true_eval, u_pred_eval = evaluate_solution_error(model, a_true)
    print(f"Regression solution relative L2 error = {sol_err:.6e}")

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Regression training loss")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t_eval, u_true_eval, label="True")
    plt.plot(t_eval, u_pred_eval, "--", label="Prediction")
    plt.scatter(t_train.numpy(), u_obs_train.numpy(), s=20, alpha=0.6, label="Train data")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Regression result")
    plt.legend()
    plt.grid(True)
    plt.show()


def demo_forward_pinn(a_true=2.0, noise_std=0.0):
    print("\n====================")
    print("Demo: Forward PINN")
    print("====================")

    t_ref, u_true_ref, u_obs_ref = generate_dataset(
        n_points=20, a=a_true, noise_std=noise_std
    )
    t_col = make_uniform_grid(100)

    model = MLP(hidden_dim=32, depth=2)
    model, history = train_forward_pinn(
        model,
        t_ref,
        u_obs_ref,
        t_col,
        a=a_true,
        epochs=3000,
        lr=1e-3,
        w_data=1.0,
        w_phys=1.0,
        verbose=True
    )

    sol_err, t_eval, u_true_eval, u_pred_eval = evaluate_solution_error(model, a_true)
    print(f"Forward PINN solution relative L2 error = {sol_err:.6e}")

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss_total"], label="total")
    plt.plot(history["loss_data"], label="data")
    plt.plot(history["loss_phys"], label="physics")
    plt.plot(history["loss_ic"], label="ic")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Forward PINN training loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t_eval, u_true_eval, label="True")
    plt.plot(t_eval, u_pred_eval, "--", label="Prediction")
    plt.scatter(t_ref.numpy(), u_obs_ref.numpy(), s=20, alpha=0.6, label="Reference data")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Forward PINN result")
    plt.legend()
    plt.grid(True)
    plt.show()


def demo_inverse_pinn(a_true=2.0, noise_std=0.0):
    print("\n====================")
    print("Demo: Inverse PINN")
    print("====================")

    t_ref, u_true_ref, u_obs_ref = generate_dataset(
        n_points=20, a=a_true, noise_std=noise_std
    )
    t_col = make_uniform_grid(100)

    model = MLP(hidden_dim=32, depth=2)
    model, a_hat, history = train_inverse_pinn(
        model,
        t_ref,
        u_obs_ref,
        t_col,
        raw_a_init=0.1,
        epochs=4000,
        lr_model=1e-3,
        lr_param=1e-3,
        w_data=1.0,
        w_phys=1.0,
        verbose=True
    )

    sol_err, t_eval, u_true_eval, u_pred_eval = evaluate_solution_error(model, a_true)
    p_err = parameter_relative_error(a_true, a_hat)

    print(f"Inverse PINN solution relative L2 error = {sol_err:.6e}")
    print(f"True a = {a_true:.6f}, Predicted a = {a_hat:.6f}")
    print(f"Parameter relative error = {p_err:.6e}")

    plt.figure(figsize=(6, 4))
    plt.plot(history["loss_total"], label="total")
    plt.plot(history["loss_data"], label="data")
    plt.plot(history["loss_phys"], label="physics")
    plt.plot(history["loss_ic"], label="ic")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Inverse PINN training loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history["a_hat"], label="a_hat")
    plt.axhline(a_true, color="r", linestyle="--", label="true a")
    plt.xlabel("Epoch")
    plt.ylabel("Estimated a")
    plt.title("Inverse PINN parameter estimation")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t_eval, u_true_eval, label="True")
    plt.plot(t_eval, u_pred_eval, "--", label="Prediction")
    plt.scatter(t_ref.numpy(), u_obs_ref.numpy(), s=20, alpha=0.6, label="Reference data")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Inverse PINN result")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    a_true = 2.0

    # --------------------------------------------------------
    # Demo runs
    # --------------------------------------------------------
    demo_regression(a_true=a_true, noise_std=0.0)
    demo_forward_pinn(a_true=a_true, noise_std=0.0)
    demo_inverse_pinn(a_true=a_true, noise_std=0.0)

    # --------------------------------------------------------
    # K-fold CV examples
    # --------------------------------------------------------
    print("\n====================")
    print("K-fold CV: Regression")
    print("====================")
    t_all, u_true_all, u_obs_all = generate_dataset(
        n_points=100, a=a_true, noise_std=0.01
    )
    cv_reg = kfold_regression(
        t_all, u_obs_all, k=5, hidden_dim=32, depth=2,
        epochs=1000, lr=1e-3, seed=0
    )
    print(cv_reg)

    print("\n====================")
    print("K-fold CV: Inverse PINN")
    print("====================")
    cv_inv = kfold_inverse_pinn(
        t_all, u_obs_all, a_true=a_true, n_collocation=100, k=5,
        hidden_dim=32, depth=2, epochs=1500,
        lr_model=1e-3, lr_param=1e-3,
        w_data=1.0, w_phys=1.0, seed=0
    )
    print(cv_inv)

    # --------------------------------------------------------
    # Requested analyses
    # --------------------------------------------------------

    # (1) Regression: train data size vs relative l2 error
    print("\n====================")
    print("Analysis (1): Regression")
    print("====================")
    res_reg = experiment_regression_by_train_size(
        train_sizes=[5, 10, 20, 50, 100],
        a_true=a_true,
        noise_std=0.0,
        hidden_dim=32,
        depth=2,
        epochs=1500,
        lr=1e-3,
        repeats=3
    )
    print(res_reg)
    plot_regression_by_train_size(res_reg)

    # (2) Forward PINN: collocation size vs relative l2 error
    print("\n====================")
    print("Analysis (2): Forward PINN")
    print("====================")
    res_fpinn = experiment_forward_pinn_by_collocation(
        n_ref=20,
        collocation_sizes=[10, 20, 50, 100, 200],
        a_true=a_true,
        noise_std=0.0,
        hidden_dim=32,
        depth=2,
        epochs=2000,
        lr=1e-3,
        w_data=1.0,
        w_phys=1.0,
        repeats=3
    )
    print(res_fpinn)
    plot_forward_pinn_by_collocation(res_fpinn)

    # (3-A) Inverse PINN: fixed reference data, varying collocation
    print("\n====================")
    print("Analysis (3-A): Inverse PINN fixed ref, vary collocation")
    print("====================")
    res_inv_col = experiment_inverse_pinn_fixed_ref_vary_col(
        n_ref=20,
        collocation_sizes=[10, 20, 50, 100, 200],
        a_true=a_true,
        noise_std=0.0,
        hidden_dim=32,
        depth=2,
        epochs=2500,
        lr_model=1e-3,
        lr_param=1e-3,
        w_data=1.0,
        w_phys=1.0,
        repeats=3
    )
    print(res_inv_col)
    plot_inverse_pinn_fixed_ref_vary_col(res_inv_col)

    # (3-B) Inverse PINN: fixed collocation, varying reference data
    print("\n====================")
    print("Analysis (3-B): Inverse PINN fixed collocation, vary reference data")
    print("====================")
    res_inv_ref = experiment_inverse_pinn_fixed_col_vary_ref(
        n_col=100,
        ref_sizes=[5, 10, 20, 50, 100],
        a_true=a_true,
        noise_std=0.0,
        hidden_dim=32,
        depth=2,
        epochs=2500,
        lr_model=1e-3,
        lr_param=1e-3,
        w_data=1.0,
        w_phys=1.0,
        repeats=3
    )
    print(res_inv_ref)
    plot_inverse_pinn_fixed_col_vary_ref(res_inv_ref)