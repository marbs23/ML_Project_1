from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


NASA_DATASET_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
RELEVANT_FILES = {"train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt"}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_dir / "CMAPSSData.zip"

    urllib.request.urlretrieve(NASA_DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            filename = Path(member).name
            if filename in RELEVANT_FILES:
                with archive.open(member) as source, (raw_data_dir / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)

    zip_path.unlink(missing_ok=True)
    

if __name__ == "__main__":
    main()

