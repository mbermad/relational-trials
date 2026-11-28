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
    train_full: Optional[Table] = None  # all training labels, when ``train`` is subsampled

    @property
    def history_rows(self) -> pd.DataFrame:
        """Labels usable as history: the FULL train split plus val (each still used only
        once known). Subsampling the training rows never shrinks the history."""
        tr = (self.train_full if self.train_full is not None else self.train).df
        return pd.concat([tr, self.val.df], ignore_index=True)

    @property
    def base_name(self) -> str:
        from pathlib import Path

        return Path(self.dataset_name).name.replace("-synthetic", "")

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


def latest_rows(table: Table, n: int) -> Table:
    """The training rows from the most recent seed times, whole seed times only, ~n rows."""
    df = table.df
    counts = df[table.time_col].value_counts().sort_index(ascending=False)
    keep = counts.index[counts.cumsum().shift(fill_value=0) < n]
    sub = df[df[table.time_col].isin(keep)].reset_index(drop=True)
    return Table(sub, table.fkey_col_to_pkey_table, table.pkey_col, table.time_col)


def load_bundle(task_name: str, dataset_name: str = DEFAULT_DATASET,
                dataset: Optional[Dataset] = None, max_train: Optional[int] = None) -> TaskBundle:
    """Load a dataset + task and all three label tables.

    ``dataset_name`` may be a bare RelBench name ("rel-trial"), a Hub spec, or a local
    folder (used by the tests).
    """
    ds = dataset if dataset is not None else relbench.load_dataset(dataset_name)
    task = ds.load_task(task_name)
    if task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        raise ValueError(f"Unsupported task type for {task_name}: {task.task_type}")
    db = task.get_db()  # hold on to it: RelBench 3 re-reads data on every call
    train = task.get_table("train")
    full = None
    if max_train and len(train.df) > max_train:
        full, train = train, latest_rows(train, max_train)
        print(f"  training on the latest {len(train.df):,} of {len(full.df):,} train rows "
              f"(from {train.df[train.time_col].min():%Y-%m-%d})")
    return TaskBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=ds,
        task=task,
        db=db,
        train=train,
        val=task.get_table("val"),
        test=task.get_table("test", mask_input_cols=False),
        train_full=full,
    )
