import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from urllib.request import urlopen


OFFICIAL_URL = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
FALLBACK_URL = "https://omnomnom.vision.rwth-aachen.de/data/point2vec/data/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
EXPECTED_SHA256 = "0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe"
ARCHIVE_NAME = "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
EXTRACTED_DIR = "shapenetcore_partanno_segmentation_benchmark_v0_normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ShapeNetPart dataset archive")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/lyx/datasets/ShapeNetPart",
        help="Directory to store the archive and extracted dataset",
    )
    parser.add_argument("--skip-sha256", action="store_true", help="Skip archive checksum verification")
    return parser.parse_args()


def sha256sum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output_path: str) -> None:
    print(f"Downloading from: {url}")
    with urlopen(url) as response, open(output_path, "wb") as f:
        shutil.copyfileobj(response, f)


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)

    archive_path = os.path.join(root, ARCHIVE_NAME)
    extracted_path = os.path.join(root, EXTRACTED_DIR)

    if not os.path.exists(archive_path):
        for url in (OFFICIAL_URL, FALLBACK_URL):
            try:
                download(url, archive_path)
                break
            except Exception as exc:
                print(f"Download failed from {url}: {exc}")
        else:
            raise RuntimeError("Failed to download ShapeNetPart from both official and fallback URLs.")

    if not args.skip_sha256:
        digest = sha256sum(archive_path)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"ShapeNetPart archive checksum mismatch.\nExpected: {EXPECTED_SHA256}\nActual:   {digest}"
            )
        print("SHA256 check passed.")

    if not os.path.isdir(extracted_path):
        print(f"Extracting to: {root}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(root)
    else:
        print(f"Extraction already exists: {extracted_path}")

    print(f"Dataset ready at: {extracted_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
