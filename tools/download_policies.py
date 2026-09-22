"""Download the exact evaluated checkpoints and standing policy without a login."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path.cwd(),
                        help="Training repository checkout; defaults to the current directory.")
    args = parser.parse_args()
    artifacts = json.loads(Path(__file__).with_name("policy_downloads.json").read_text())
    for artifact in artifacts:
        destination = args.training_root / artifact["destination"]
        if destination.exists():
            if sha256(destination) != artifact["sha256"]:
                raise RuntimeError(f"Existing file has a different hash; left unchanged: {destination}")
            print(f"Verified existing {destination}")
            continue
        downloaded = Path(hf_hub_download(
            repo_id=artifact["repository"], filename=artifact["filename"],
            revision=artifact["revision"], token=False,
        ))
        if sha256(downloaded) != artifact["sha256"]:
            raise RuntimeError(f"Downloaded file has a different hash: {artifact['repository']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, destination)
        print(f"Downloaded and verified {destination}")


if __name__ == "__main__":
    main()
