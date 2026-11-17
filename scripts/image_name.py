#!/usr/bin/env python3
"""Resolve this branch's mutable TensorFold image tag without touching immutable digests."""

from __future__ import annotations

import argparse

DEFAULT_IMAGE = "glm53-tensorfold:dev"
FORK_SUFFIX = "-0rand"


def fork_image(image: str = "") -> str:
    """Return the branch-owned tag for *image*; digest-pinned references are immutable."""
    image = image or DEFAULT_IMAGE
    if "@" in image:
        return image

    tag_start = image.rfind(":")
    if tag_start <= image.rfind("/"):
        return f"{image}:latest{FORK_SUFFIX}"
    if image.endswith(FORK_SUFFIX):
        return image
    return f"{image}{FORK_SUFFIX}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="", help=f"image reference (default: {DEFAULT_IMAGE})")
    print(fork_image(parser.parse_args().image))


if __name__ == "__main__":
    main()
