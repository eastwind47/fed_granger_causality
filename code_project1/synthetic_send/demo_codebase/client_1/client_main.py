from pathlib import Path

from client_runner import run_client


if __name__ == "__main__":
    run_client(Path(__file__).with_name("client1_config.json"))
