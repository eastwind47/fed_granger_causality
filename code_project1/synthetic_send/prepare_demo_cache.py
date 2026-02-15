import numpy as np

from demo_common import (
    amn_path,
    client_state_dir,
    ensure_dir,
    load_component_data,
    load_config,
    metadata_path,
    resolve_path,
    runtime_templates,
    save_matrix,
    server_state_dir,
    write_gradient_message,
    write_json,
)
from local_learner import LocalModel


def main() -> None:
    cfg = load_config()

    data_dir = resolve_path(cfg, cfg["paths"]["data_dir"])
    state_dir = resolve_path(cfg, cfg["paths"]["state_dir"])
    client_input_dir = resolve_path(cfg, cfg["paths"]["client_input_dir"])

    ensure_dir(state_dir)
    ensure_dir(client_input_dir)

    m_total = int(cfg["system"]["num_clients"])
    eta_l = cfg["local"]["eta_l"]
    eta_g = cfg["local"]["eta_g"]
    lambda_l = cfg["local"]["lambda_l"]
    if not (len(eta_l) == len(eta_g) == len(lambda_l) == m_total):
        raise ValueError("Length of local hyperparameter vectors must match num_clients.")

    theta_rng = np.random.default_rng(int(cfg["initialization"]["theta_seed"]))
    theta_mean = float(cfg["initialization"]["theta_mean"])
    theta_std = float(cfg["initialization"]["theta_std"])

    p_dims = []
    d_dims = []
    training_time = None

    for m in range(1, m_total + 1):
        comp = load_component_data(
            data_dir=data_dir,
            client_id=m,
            eta_l=eta_l[m - 1],
            eta_g=eta_g[m - 1],
            lambda_l=lambda_l[m - 1],
            force_zero_b=True,
        )
        p_m = int(comp["p_m"])
        d_m = int(comp["d_m"])
        t_horizon = int(comp["Y"].shape[1])

        if training_time is None:
            training_time = t_horizon
        elif training_time != t_horizon:
            raise ValueError(f"Inconsistent Y horizon across clients. client_{m} has T={t_horizon}")

        theta_0 = theta_rng.normal(theta_mean, theta_std, size=(p_m, d_m))

        local_model = LocalModel(training_time, comp, theta_0.copy())
        local_model.run_DKF()

        c_state_dir = client_state_dir(state_dir, m)
        ensure_dir(c_state_dir)

        save_matrix(c_state_dir / "A.csv", comp["A"])
        save_matrix(c_state_dir / "C.csv", comp["C"])
        save_matrix(c_state_dir / "Y.csv", comp["Y"])
        save_matrix(c_state_dir / "x0.csv", comp["x0"])
        save_matrix(c_state_dir / "X_dkf.csv", local_model.X_dkf)
        save_matrix(c_state_dir / "X_dkf_pred.csv", local_model.X_dkf_pred)
        save_matrix(c_state_dir / "theta.csv", theta_0)
        save_matrix(c_state_dir / "X_vfl_history.csv", np.zeros((p_m, training_time)))

        p_dims.append(p_m)
        d_dims.append(d_m)

    global_cfg = {
        "gamma_g": float(cfg["global"]["gamma_g"]),
        "lambda_g": float(cfg["global"]["lambda_g"]),
    }
    write_json(server_state_dir(state_dir) / "global_params.json", global_cfg)

    amn_rng = np.random.default_rng(int(cfg["initialization"]["amn_seed"]))
    amn_mean = float(cfg["initialization"]["amn_mean"])
    amn_std = float(cfg["initialization"]["amn_std"])
    ensure_dir(server_state_dir(state_dir))
    for m in range(1, m_total + 1):
        for n in range(1, m_total + 1):
            if m == n:
                continue
            amn_block = amn_rng.normal(amn_mean, amn_std, size=(p_dims[m - 1], p_dims[n - 1]))
            save_matrix(amn_path(state_dir, m, n), amn_block)

    md = {
        "num_clients": m_total,
        "training_time": int(training_time),
        "p_dims": p_dims,
        "d_dims": d_dims,
    }
    write_json(metadata_path(state_dir), md)

    grad_template, _ = runtime_templates(cfg)
    for m in range(1, m_total + 1):
        grad_path = client_input_dir / grad_template.format(client_id=m)
        write_gradient_message(grad_path, t=0, grad=np.zeros((p_dims[m - 1], 1)))

    print(f"Prepared demo cache at {state_dir}")
    print(f"Bootstrap gradients written to {client_input_dir}")


if __name__ == "__main__":
    main()

