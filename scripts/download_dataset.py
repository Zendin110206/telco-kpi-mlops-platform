from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

ZENODO_RECORD_URL = "https://zenodo.org/records/8147768"
DOWNLOAD_URL = (
    "https://zenodo.org/records/8147768/files/"
    "network_operator_KPIs_time_series_dataset.zip?download=1"
)
REMOTE_FILE_NAME = "network_operator_KPIs_time_series_dataset.zip"
LOCAL_ARCHIVE_NAME = "network_operator_kpis_time_series_dataset.zip"
EXPECTED_MD5 = "8a3bbc403a99a7c175d3a0703a5ca8fe"

RAW_DATA_DIR = Path("data/raw")
EXTRACT_DIR = RAW_DATA_DIR / "network_operator_kpis"
ARCHIVE_PATH = RAW_DATA_DIR / LOCAL_ARCHIVE_NAME


def calculate_md5(path: Path) -> str:
    """Calculate the MD5 hash of a file."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a specified destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def verify_checksum(path: Path, expected_md5: str) -> None:
    """Verify the MD5 checksum of a file."""
    actual_md5 = calculate_md5(path)

    if actual_md5 != expected_md5:
        raise ValueError(
            f"Downloaded file checksum mismatch. Expected: {expected_md5}, got {actual_md5}"
        )


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive to a specified directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)


def main(force: bool) -> None:
    print("Dataset source:")
    print(f"- Record URL: {ZENODO_RECORD_URL}")
    print(f"- Remote file: {REMOTE_FILE_NAME}")
    print(f"- Local archive: {ARCHIVE_PATH}")

    if ARCHIVE_PATH.exists() and not force:
        print("Archive already exists. Verifying checksum...")
    else:
        print("Downloading dataset archive...")
        download_file(DOWNLOAD_URL, ARCHIVE_PATH)

    verify_checksum(ARCHIVE_PATH, EXPECTED_MD5)
    print("MD5 checksum OK.")

    print(f"Extracting archive to {EXTRACT_DIR}...")
    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)

    extracted_files = sorted(path for path in EXTRACT_DIR.rglob("*") if path.is_file())
    print(f"Extracted files: {len(extracted_files)}")
    for path in extracted_files[:10]:
        print(f"- {path}")

    if len(extracted_files) > 10:
        print(f"... and {len(extracted_files) - 10} more files.")

    print("Dataset download step completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract the network operator KPIs time series dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the archive again even if it already exists.",
    )
    args = parser.parse_args()

    main(force=args.force)
