#!/usr/bin/env python3
"""
Download and extract the example dataset from the GitHub release.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


def download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def extract_zip(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.5.0", help="Release tag (default: v0.5.0)")
    parser.add_argument(
        "--asset",
        default="droute-data-0.5.0.zip",
        help="Asset name (default: droute-data-0.5.0.zip)",
    )
    parser.add_argument(
        "--dest",
        default="data",
        help="Destination directory (default: data)",
    )
    args = parser.parse_args()

    url = (
        "https://github.com/DarriEy/dRoute/releases/download/"
        f"{args.tag}/{args.asset}"
    )
    dest = Path(args.dest)

    print(f"Downloading {url} ...")
    try:
        payload = download(url)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    try:
        extract_zip(payload, dest)
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Data extracted to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
