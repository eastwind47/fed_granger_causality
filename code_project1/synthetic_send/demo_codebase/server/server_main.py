import argparse
import pickle
import socket
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data_retriever1 import build_startup_bundles, load_server_config, persist_server_bundle
from global_learner import GlobalModel


HEADER_BYTES = 8


def send_msg(sock: socket.socket, payload: Dict) -> None:
    data = pickle.dumps(payload)
    sock.sendall(len(data).to_bytes(HEADER_BYTES, "big"))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket connection closed while receiving data.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> Dict:
    size = int.from_bytes(_recv_exact(sock, HEADER_BYTES), "big")
    payload = _recv_exact(sock, size)
    obj = pickle.loads(payload)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload, received {type(obj).__name__}.")
    return obj


def _col_vector(values, size: int, field_name: str, client_id: int, round_idx: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != size:
        raise ValueError(
            f"Invalid {field_name} size from client_{client_id} at round {round_idx}: "
            f"received {arr.size}, expected {size}."
        )
    return arr.reshape(size, 1)


def _expect_message_type(msg: Dict, expected: str, context: str) -> None:
    msg_type = msg.get("type")
    if msg_type != expected:
        raise ValueError(f"{context}: expected message type '{expected}', received '{msg_type}'.")


def accept_clients(
    server_sock: socket.socket,
    num_clients: int,
    recv_timeout_sec: float,
) -> Dict[int, Tuple[socket.socket, Tuple[str, int]]]:
    clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]] = {}

    while len(clients) < num_clients:
        conn, addr = server_sock.accept()
        conn.settimeout(recv_timeout_sec)

        hello = recv_msg(conn)
        _expect_message_type(hello, "hello", f"connection from {addr}")

        client_id = int(hello.get("client_id", -1))
        if not (1 <= client_id <= num_clients):
            raise ValueError(f"Received invalid client_id={client_id} from {addr}.")
        if client_id in clients:
            raise ValueError(f"Duplicate connection for client_id={client_id}.")

        clients[client_id] = (conn, addr)
        print(f"server: client_{client_id} connected from {addr}")

    return clients


def send_bootstrap(clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]], server_bundle: Dict, client_bundles: Dict[int, Dict]) -> None:
    metadata = server_bundle["metadata"]
    global_params = server_bundle["global"]

    for client_id in sorted(clients):
        conn, _ = clients[client_id]
        msg = {
            "type": "bootstrap",
            "round": 0,
            "client_id": client_id,
            "metadata": metadata,
            "global": global_params,
            "bundle": client_bundles[client_id],
        }
        send_msg(conn, msg)

    for client_id in sorted(clients):
        conn, _ = clients[client_id]
        ack = recv_msg(conn)
        _expect_message_type(ack, "bootstrap_ack", f"client_{client_id} bootstrap ack")

        ack_client_id = int(ack.get("client_id", -1))
        status = str(ack.get("status", ""))
        if ack_client_id != client_id:
            raise ValueError(
                f"Bootstrap ack client mismatch: expected client_{client_id}, "
                f"received client_{ack_client_id}."
            )
        if status.lower() not in {"ok", "ready"}:
            raise ValueError(
                f"Bootstrap ack from client_{client_id} returned non-success status='{status}'."
            )
        print(f"server: bootstrap acknowledged by client_{client_id}")


def run_synchronized_rounds(clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]], server_bundle: Dict) -> Dict:
    metadata = server_bundle["metadata"]
    num_clients = int(metadata["num_clients"])
    num_rounds = int(metadata["num_rounds"])
    p_dims = [int(x) for x in metadata["p_dims"]]

    gl_data = {
        "A_mm": server_bundle["A_mm"],
        "gamma_g": float(server_bundle["global"]["gamma_g"]),
        "lambda_g": float(server_bundle["global"]["lambda_g"]),
    }
    global_model = GlobalModel(num_clients, num_rounds, gl_data, server_bundle["A_mn"])

    grad_x = {f"{m}": np.zeros((p_dims[m - 1], 1)) for m in range(1, num_clients + 1)}

    for t in range(num_rounds):
        for client_id in sorted(clients):
            conn, _ = clients[client_id]
            send_msg(
                conn,
                {
                    "type": "gradient",
                    "round": t,
                    "client_id": client_id,
                    "grad_x": grad_x[f"{client_id}"],
                },
            )

        x_dkf = {}
        x_vfl = {}

        for client_id in sorted(clients):
            conn, _ = clients[client_id]
            msg = recv_msg(conn)
            _expect_message_type(msg, "update", f"client_{client_id} round {t}")

            round_idx = int(msg.get("round", -1))
            msg_client_id = int(msg.get("client_id", -1))
            if round_idx != t:
                raise ValueError(
                    f"Round mismatch from client_{client_id}: expected {t}, received {round_idx}."
                )
            if msg_client_id != client_id:
                raise ValueError(
                    f"Client mismatch at round {t}: expected client_{client_id}, "
                    f"received client_{msg_client_id}."
                )

            p_m = p_dims[client_id - 1]
            x_dkf[f"{client_id}"] = _col_vector(msg.get("x_dkf"), p_m, "x_dkf", client_id, t)
            x_vfl[f"{client_id}"] = _col_vector(msg.get("x_vfl"), p_m, "x_vfl", client_id, t)

        current_loss = float(global_model.GlobalLoss(x_dkf, x_vfl, t))
        grad_x = global_model.Gradx(x_dkf, x_vfl, t)
        server_bundle["A_mn"] = global_model.GradDescent(x_dkf, x_vfl, t)

        print(f"server: completed round {t + 1}/{num_rounds}, global_loss={current_loss:.6f}")

    return server_bundle


def send_stop(clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]], final_round: int) -> None:
    for client_id in sorted(clients):
        conn, _ = clients[client_id]
        send_msg(conn, {"type": "stop", "round": final_round, "client_id": client_id})


def close_all(clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]]) -> None:
    for conn, _ in clients.values():
        try:
            conn.close()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Socket server for federated Granger demo.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("server_config.json")),
        help="Path to server config JSON.",
    )
    args = parser.parse_args()

    cfg = load_server_config(Path(args.config))
    server_bundle, client_bundles = build_startup_bundles(cfg)
    persist_server_bundle(cfg, server_bundle)

    network_cfg = cfg["network"]
    host = str(network_cfg.get("host", "0.0.0.0"))
    port = int(network_cfg["port"])
    backlog = int(network_cfg.get("backlog", 8))
    recv_timeout_sec = float(network_cfg.get("recv_timeout_sec", 120))
    num_clients = int(server_bundle["metadata"]["num_clients"])

    clients: Dict[int, Tuple[socket.socket, Tuple[str, int]]] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(backlog)
        print(f"server: listening on {host}:{port}, expecting {num_clients} clients")

        try:
            clients = accept_clients(server_sock, num_clients, recv_timeout_sec)
            send_bootstrap(clients, server_bundle, client_bundles)
            server_bundle = run_synchronized_rounds(clients, server_bundle)

            if bool(cfg.get("runtime", {}).get("send_stop_message", True)):
                send_stop(clients, int(server_bundle["metadata"]["num_rounds"]))

            persist_server_bundle(cfg, server_bundle)
            print("server: training complete")
        finally:
            close_all(clients)


if __name__ == "__main__":
    main()
