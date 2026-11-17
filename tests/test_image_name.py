"""Image-reference policy for the main-0rand TensorFold fork."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from image_name import fork_image  # noqa: E402

@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("", "glm53-tensorfold:dev-0rand"),
        ("glm53-tensorfold:dev", "glm53-tensorfold:dev-0rand"),
        ("glm53-tensorfold", "glm53-tensorfold:latest-0rand"),
        ("registry:5000/glm53-tensorfold", "registry:5000/glm53-tensorfold:latest-0rand"),
        ("registry:5000/glm53-tensorfold:green", "registry:5000/glm53-tensorfold:green-0rand"),
        ("registry:5000/glm53-tensorfold:green-0rand", "registry:5000/glm53-tensorfold:green-0rand"),
        ("registry:5000/glm53-tensorfold@sha256:deadbeef", "registry:5000/glm53-tensorfold@sha256:deadbeef"),
    ],
)
def test_fork_image(image, expected):
    assert fork_image(image) == expected


def test_image_name_cli_defaults_and_accepts_a_custom_tag():
    script = ROOT / "scripts" / "image_name.py"
    assert subprocess.run([sys.executable, str(script)], text=True, capture_output=True, check=True).stdout.strip() == \
           "glm53-tensorfold:dev-0rand"
    assert subprocess.run([sys.executable, str(script), "custom:tag"], text=True, capture_output=True, check=True).stdout.strip() == \
           "custom:tag-0rand"
