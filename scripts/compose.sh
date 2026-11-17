#!/usr/bin/env bash
# Run the debugging compose stack with a branch-specific image without altering the configured base image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${CONFIG:-config/minimal.env}"
if [[ ! -f "$CONFIG" ]]; then
    if [[ "$CONFIG" == config/minimal.env ]]; then
        echo "[glm53-tf] no config/minimal.env. Create it with: cp config/minimal.env.example config/minimal.env" >&2
    else
        echo "[glm53-tf] no config file $CONFIG. Set CONFIG to an existing compose configuration file." >&2
    fi
    exit 2
fi

# Compose gives shell variables precedence over --env-file. Keep that behavior for every setting: only inspect the
# trusted config in a subshell to select IMAGE when the caller did not provide one.
if [[ ${IMAGE+x} ]]; then
    source_image="$IMAGE"
else
    if ! source_image=$(unset IMAGE; set -a; source "$CONFIG" && printf '%s' "${IMAGE:-}"); then
        echo "[glm53-tf] cannot read IMAGE from config file $CONFIG." >&2
        exit 2
    fi
fi
TF_IMAGE=$(python3 scripts/image_name.py "$source_image")
export TF_IMAGE

exec docker compose --env-file "$CONFIG" -f docker/compose.yaml "$@"
