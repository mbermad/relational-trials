"""Loading rel-trial tasks from RelBench (v3) into one convenient bundle.

RelBench 3 loads datasets straight from the Hugging Face Hub (or a local folder with the
same layout, which is what the synthetic test dataset uses). Test labels are hidden by
default; for offline evaluation we read them with ``mask_input_cols=False``, which is
exactly what ``relbench.submit.evaluate_task`` does internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

import relbench
from relbench.base import Database, Dataset, EntityTask, Table, TaskType

DEFAULT_DATASET = "rel-trial"
TASKS = ("study-outcome", "study-adverse")


@dataclass
class TaskBundle:
    """Everything one experiment needs, loaded once (RelBench 3 does no caching)."""

    dataset_name: str
    task_name: str
    dataset: Dataset
    task: EntityTask
    db: Database  # the task's view of the database (hidden columns removed)
    train: Table
    val: Table
    test: Table  # WITH labels, for offline evaluation

    @property
    def is_classification(self) -> bool:
        return self.task.task_type == TaskType.BINARY_CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        return self.task.task_type == TaskType.REGRESSION

    @property
    def entity_df(self) -> pd.DataFrame:
        return self.db.table_dict[self.task.entity_table].df

    def split(self, name: str) -> Table:
        return {"train": self.train, "val": self.val, "test": self.test}[name]

    def nmae_std(self) -> float:
        """Normalizer RelBench uses for NMAE (train-split target std, ddof=1)."""
        std = getattr(self.task, "nmae_std", None)
        if std is None:
            std = float(self.train.df[self.task.target_col].std(ddof=1))
        return float(std)


def load_bundle(task_name: str, dataset_name: str = DEFAULT_DATASET,
                dataset: Optional[Dataset] = None) -> TaskBundle:
    """Load a dataset + task and all three label tables.

    ``dataset_name`` may be a bare RelBench name ("rel-trial"), a Hub spec, or a local
    folder (used by the tests).
    """
    ds = dataset if dataset is not None else relbench.load_dataset(dataset_name)
    task = ds.load_task(task_name)
    if task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        raise ValueError(f"Unsupported task type for {task_name}: {task.task_type}")
    db = task.get_db()  # hold on to it: RelBench 3 re-reads data on every call
    return TaskBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=ds,
        task=task,
        db=db,
        train=task.get_table("train"),
        val=task.get_table("val"),
        test=task.get_table("test", mask_input_cols=False),
    )
