from typing import Callable, Optional
import os

from .envs import STORAGE_PATH
from .models import db as flask_db, TestDataset

DATASET_PATH = os.path.join(STORAGE_PATH, "test_datasets")


def get_test_datasets() -> list[tuple[str, str, str]]:
    datasets: list[TestDataset] = TestDataset.query.all()
    return [(ds.name, ds.description, ds.file_path) for ds in datasets]


def upload_test_dataset(
    file_saver: Callable[[str], None],
    ds_name: str,
    ds_description: str,
    overwrite: bool,
) -> Optional[str]:
    # Save the uploaded file to the desired location
    file_path = os.path.join(DATASET_PATH, ds_name)

    if os.path.exists(file_path) and not overwrite:
        return f"Dataset file already exists: {ds_name}"

    # save on disk
    file_saver(file_path)

    # save info to databse
    descr = TestDataset(name=ds_name, description=ds_description,
                        file_path=file_path)

    flask_db.session.add(descr)
    flask_db.session.commit()
