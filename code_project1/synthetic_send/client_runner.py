from pathlib import Path

import numpy as np

from demo_common import (
    client_state_dir,
    ensure_dir,
    load_config,
    load_matrix,
    metadata_path,
    read_client_state_message,
    read_gradient_message,
    read_json,
    resolve_path,
    runtime_templates,
    save_matrix,
    write_client_state_message,
)
from local_learner import LocalModel


def _resolve_io_dir(cfg, override: str, cfg_key: str) -> Path:
    if override:
        return Path(override).resolve()
    return resolve_path(cfg, cfg["paths"][cfg_key])


def run_client(client_id: int, input_loc_override: str = "", output_loc_override: str = "") -> None:
    cfg = load_config()
    state_dir = resolve_path(cfg, cfg["paths"]["state_dir"])
    md = read_json(metadata_path(state_dir))

    m_total = int(md["num_clients"])
    if not (1 <= client_id <= m_total):
        raise ValueError(f"client_id={client_id} is out of range for num_clients={m_total}")

    training_time = int(md["training_time"])
    p_m = int(md["p_dims"][client_id - 1])
    d_m = int(md["d_dims"][client_id - 1])

    grad_template, state_template = runtime_templates(cfg)
    input_loc = _resolve_io_dir(cfg, input_loc_override, "client_input_dir")
    output_loc = _resolve_io_dir(cfg, output_loc_override, "client_output_dir")
    ensure_dir(input_loc)
    ensure_dir(output_loc)

    grad_path = input_loc / grad_template.format(client_id=client_id)
    if not grad_path.exists():
        raise FileNotFoundError(f"Gradient input file not found: {grad_path}")
    t, grad_x = read_gradient_message(grad_path, p_m)
    if t < 0 or t >= training_time:
        raise ValueError(f"Received t={t}, but valid range is [0, {training_time - 1}]")

    c_state = client_state_dir(state_dir, client_id)
    a = load_matrix(c_state / "A.csv")
    c = load_matrix(c_state / "C.csv")
    y = load_matrix(c_state / "Y.csv")
    x0 = load_matrix(c_state / "x0.csv").reshape(-1, 1)
    x_dkf = load_matrix(c_state / "X_dkf.csv")
    theta = load_matrix(c_state / "theta.csv")
    x_vfl_hist = load_matrix(c_state / "X_vfl_history.csv")

    # Maintain compatibility with LocalModel input contract.
    b = np.zeros((p_m, p_m))
    q = 0.0005 * np.eye(p_m)
    r = 0.0005 * np.eye(d_m)
    comp_data = {
        "A": a,
        "B": b,
        "C": c,
        "Q": q,
        "R": r,
        "P0": q.copy(),
        "x0": x0,
        "Y": y,
        "d_m": d_m,
        "p_m": p_m,
        "eta_l": float(cfg["local"]["eta_l"][client_id - 1]),
        "eta_g": float(cfg["local"]["eta_g"][client_id - 1]),
        "lambda_l": float(cfg["local"]["lambda_l"][client_id - 1]),
    }

    model = LocalModel(training_time, comp_data, theta.copy())
    model.X_dkf = x_dkf.copy()
    model.X_vfl = x_vfl_hist.copy()

    x_vfl_t = model.VFL_estimate(t)
    x_vfl_pred_t = model.VFL_prediction(t)
    x_dkf_t = model.X_dkf[:, t : t + 1]

    model.GradDescent(grad_x, t)

    x_vfl_hist[:, t : t + 1] = x_vfl_t
    save_matrix(c_state / "X_vfl_history.csv", x_vfl_hist)
    save_matrix(c_state / "theta.csv", model.theta)

    state_out_path = output_loc / state_template.format(client_id=client_id)
    write_client_state_message(
        state_out_path,
        t=t,
        x_dkf=x_dkf_t,
        x_vfl=x_vfl_t,
        x_vfl_pred=x_vfl_pred_t,
    )

    print(
        f"client_{client_id}: consumed {grad_path.name} at t={t}, "
        f"wrote {state_out_path.name}"
    )


def preview_client_state_message(path: Path, p_m: int) -> None:
    t, x_dkf, x_vfl, x_vfl_pred = read_client_state_message(path, p_m)
    print("t:", t)
    print("x_dkf:", x_dkf.ravel())
    print("x_vfl:", x_vfl.ravel())
    print("x_vfl_pred:", x_vfl_pred.ravel())

