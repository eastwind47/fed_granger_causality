from pathlib import Path

from demo_common import (
    amn_path,
    client_state_dir,
    ensure_dir,
    load_config,
    load_matrix,
    metadata_path,
    read_client_state_message,
    read_json,
    resolve_path,
    runtime_templates,
    save_matrix,
    write_gradient_message,
)
from global_learner import GlobalModel

# Optional path overrides.
# Leave empty to use paths from config.json.
input_loc = ""
output_loc = ""


def _resolve_io_dir(cfg, override: str, cfg_key: str) -> Path:
    if override:
        return Path(override).resolve()
    return resolve_path(cfg, cfg["paths"][cfg_key])


def main() -> None:
    cfg = load_config()

    state_dir = resolve_path(cfg, cfg["paths"]["state_dir"])
    md = read_json(metadata_path(state_dir))

    m_total = int(md["num_clients"])
    p_dims = [int(x) for x in md["p_dims"]]
    training_time = int(md["training_time"])

    _, state_template = runtime_templates(cfg)
    grad_template, _ = runtime_templates(cfg)
    server_input_dir = _resolve_io_dir(cfg, input_loc, "server_input_dir")
    server_output_dir = _resolve_io_dir(cfg, output_loc, "server_output_dir")
    ensure_dir(server_input_dir)
    ensure_dir(server_output_dir)

    x_dkf = {}
    x_vfl = {}
    t_values = []
    for m in range(1, m_total + 1):
        state_path = server_input_dir / state_template.format(client_id=m)
        if not state_path.exists():
            raise FileNotFoundError(f"Missing client state file: {state_path}")
        t_m, x_dkf_m, x_vfl_m, _ = read_client_state_message(state_path, p_dims[m - 1])
        t_values.append(t_m)
        x_dkf[f"{m}"] = x_dkf_m
        x_vfl[f"{m}"] = x_vfl_m

    if len(set(t_values)) != 1:
        raise ValueError(f"Clients are not synchronized. Received t values: {t_values}")
    t = int(t_values[0])
    if t < 0 or t >= training_time:
        raise ValueError(f"Received t={t}, but valid range is [0, {training_time - 1}]")

    a_mm = {}
    for m in range(1, m_total + 1):
        a_mm[f"{m}{m}"] = load_matrix(client_state_dir(state_dir, m) / "A.csv")

    a_mn = {}
    for m in range(1, m_total + 1):
        for n in range(1, m_total + 1):
            if m == n:
                continue
            a_mn[f"{m}{n}"] = load_matrix(amn_path(state_dir, m, n))

    gl_data = {
        "A_mm": a_mm,
        "gamma_g": float(cfg["global"]["gamma_g"]),
        "lambda_g": float(cfg["global"]["lambda_g"]),
    }
    global_model = GlobalModel(m_total, training_time, gl_data, a_mn)

    grad_x = global_model.Gradx(x_dkf, x_vfl, t)
    updated_a_mn = global_model.GradDescent(x_dkf, x_vfl, t)

    for m in range(1, m_total + 1):
        for n in range(1, m_total + 1):
            if m == n:
                continue
            block = updated_a_mn[f"{m}{n}"]
            save_matrix(amn_path(state_dir, m, n), block)
            save_matrix(server_output_dir / f"A_mn_{m}{n}.csv", block)

    emit_next_t = bool(cfg.get("runtime", {}).get("server_emit_next_t", True))
    t_out = min(t + 1, training_time - 1) if emit_next_t else t
    for m in range(1, m_total + 1):
        grad_path = server_output_dir / grad_template.format(client_id=m)
        write_gradient_message(grad_path, t=t_out, grad=grad_x[f"{m}"])

    print(
        f"server: consumed client states at t={t}, updated A_mn, "
        f"wrote gradients for t={t_out} to {server_output_dir}"
    )


if __name__ == "__main__":
    main()

