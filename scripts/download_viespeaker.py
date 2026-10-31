from pathlib import Path

import httpx
from huggingface_hub import HfApi, snapshot_download, set_client_factory


set_client_factory(
    lambda: httpx.Client(
        headers={"Accept-Encoding": "gzip"},
        follow_redirects=True,
        timeout=None,
    )
)




REPO_ID = "hustep-lab/VieSpeaker-Dataset"

DESTINATION = Path(
    "/media/ml4u/Samsung_T51/ptnk_team/"
    "mortyhuaphuoc/datasets/VieSpeaker/repository"
)

DESTINATION.mkdir(parents=True, exist_ok=True)

print("Reading repository information...")
info = HfApi().dataset_info(REPO_ID)
revision = info.sha

print("Pinned revision:", revision)
(DESTINATION / "DOWNLOADED_REVISION.txt").write_text(
    revision + "\n",
    encoding="utf-8",
)

print("Downloading VieSpeaker repository...")
result = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    revision=revision,
    local_dir=DESTINATION,
    allow_patterns=[
        "data/*.parquet",
        "trials/*.txt",
        "README.md",
    ],
    max_workers=1,
)

print("Download completed.")
print("Revision:", revision)
print("Location:", result)
