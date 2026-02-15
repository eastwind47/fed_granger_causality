from client_runner import run_client

# Optional path overrides.
# Leave empty to use paths from config.json.
input_loc = ""
output_loc = ""


if __name__ == "__main__":
    run_client(client_id=1, input_loc_override=input_loc, output_loc_override=output_loc)

