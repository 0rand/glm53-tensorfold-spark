"""Read-only Docker Compose checks for scripts/compose.sh image isolation.

Requires the Docker Compose plugin but never contacts the daemon: ``config --format json`` only resolves the model.
Run with: ``python -m pytest -q tests/test_compose_image.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ("docker", "compose")


def _compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run([*COMPOSE, "version"], text=True, capture_output=True, timeout=10).returncode == 0
    except OSError:
        return False


needs_compose = pytest.mark.skipif(not _compose_available(), reason="needs the Docker Compose plugin")


def _write_config(path: Path, image: str) -> None:
    path.write_text(
        f"IMAGE={image}\n"
        "HEAD_IP=203.0.113.10\n"
        "NCCL_SOCKET_IFNAME=eth9\n"
        "NCCL_IB_HCA=fakehca0\n"
        "MODEL_PATH=/model\n"
        "HEAD_HF=/tmp/head-hf\n"
        "WORKER_HF=/tmp/worker-hf\n"
    )


def _clean_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("IMAGE", None)
    env.pop("TF_IMAGE", None)
    env.update(extra)
    return env


def _wrapper_config(config: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / "scripts/compose.sh"), "--profile", "rank0", "--profile", "rank1", "config", "--format", "json"],
        cwd=ROOT,
        env=dict(env, CONFIG=str(config)),
        text=True,
        capture_output=True,
        timeout=30,
    )


@needs_compose
def test_wrapper_uses_config_image_and_overrides_tf_image(tmp_path):
    config = tmp_path / "compose.env"
    _write_config(config, "registry.example:5000/acme/tensorfold:dev")

    result = _wrapper_config(config, _clean_env(TF_IMAGE="must-not-be-used:latest"))

    assert result.returncode == 0, result.stderr
    model = json.loads(result.stdout)
    assert model["services"]["rank0"]["image"] == "registry.example:5000/acme/tensorfold:dev-0rand"
    assert model["services"]["rank1"]["image"] == "registry.example:5000/acme/tensorfold:dev-0rand"


@needs_compose
def test_wrapper_caller_image_and_other_shell_values_take_precedence(tmp_path):
    config = tmp_path / "compose.env"
    _write_config(config, "config-image:old")

    result = _wrapper_config(
        config,
        _clean_env(IMAGE="caller-image:release", TF_IMAGE="must-not-be-used:latest", HEAD_IP="203.0.113.9"),
    )

    assert result.returncode == 0, result.stderr
    rank0 = json.loads(result.stdout)["services"]["rank0"]
    assert rank0["image"] == "caller-image:release-0rand"
    assert rank0["environment"]["MASTER"] == "203.0.113.9"


@needs_compose
def test_direct_compose_rejects_legacy_image_config(tmp_path):
    config = tmp_path / "compose.env"
    _write_config(config, "glm53-tensorfold:dev")

    result = subprocess.run(
        [*COMPOSE, "--env-file", str(config), "-f", "docker/compose.yaml",
         "--profile", "rank0", "--profile", "rank1", "config", "--format", "json"],
        cwd=ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode != 0
    wrapped = _wrapper_config(config, _clean_env())
    assert wrapped.returncode == 0, wrapped.stderr
    assert all(service["image"] == "glm53-tensorfold:dev-0rand"
               for service in json.loads(wrapped.stdout)["services"].values())


def test_wrapper_refuses_missing_config(tmp_path):
    config = tmp_path / "missing.env"
    result = subprocess.run(
        [str(ROOT / "scripts/compose.sh"), "config"],
        cwd=ROOT,
        env=dict(_clean_env(), CONFIG=str(config)),
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 2
