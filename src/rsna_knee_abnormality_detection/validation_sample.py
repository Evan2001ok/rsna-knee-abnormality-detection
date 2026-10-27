"""Small monitoring checks that leave the training run's state unchanged."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Integral

import numpy as np
import pandas as pd
from torch import nn

from .constants import STUDY_ID, TARGETS
from .utils import restore_rng, rng_state


def select_validation_sample(validation: pd.DataFrame, max_studies: int, seed: int) -> pd.DataFrame:
    """Select a repeatable, coverage-biased sample for monitoring, not final scoring.

    Only the supplied validation rows are eligible. The greedy first pass covers
    as many observed finding/class pairs as possible; missing and soft targets
    do not count as binary classes. Each step covers at least one remaining pair,
    so all available pairs fit in at most 24 studies when the limit permits it.
    A local seeded generator breaks ties and fills unused places randomly.

    The caller normally sets ``max_studies=32``. Zero returns every validation
    study. Returned rows are sorted by exact study ID with a fresh index; their
    columns and values are preserved. No global random state or input is changed.
    """
    if isinstance(max_studies, bool) or not isinstance(max_studies, Integral) or max_studies < 0:
        raise ValueError("max_studies must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    missing = [column for column in [STUDY_ID, *TARGETS] if column not in validation.columns]
    if missing:
        raise ValueError(f"Validation rows are missing required columns: {missing}")
    if validation.columns.duplicated().any():
        raise ValueError("Validation columns must be unique")
    ids = validation[STUDY_ID]
    keys = ids.astype(str)
    if ids.isna().any() or keys.str.strip().eq("").any():
        raise ValueError("Validation study IDs must not be blank")
    if keys.duplicated().any():
        raise ValueError("Validation study IDs must be unique")
    if keys.ne(keys.str.strip()).any():
        raise ValueError("Validation study IDs must not contain surrounding whitespace")
    ordered = validation.iloc[np.argsort(keys.to_numpy(), kind="stable")].reset_index(drop=True)
    if max_studies == 0 or len(ordered) <= max_studies:
        return ordered.copy(deep=True)

    labels = ordered[TARGETS]
    coverage = np.concatenate([
        labels.eq(value).fillna(False).to_numpy(dtype=bool) for value in (0, 1)
    ], axis=1)
    uncovered = coverage.any(axis=0)
    available = np.ones(len(ordered), dtype=bool)
    selected = []
    generator = np.random.default_rng(int(seed))
    while uncovered.any() and len(selected) < max_studies:
        gains = coverage[:, uncovered].sum(axis=1)
        gains[~available] = -1
        candidates = np.flatnonzero(gains == gains.max())
        index = int(generator.choice(candidates))
        selected.append(index)
        available[index] = False
        uncovered &= ~coverage[index]
    if len(selected) < max_studies:
        selected.extend(generator.choice(
            np.flatnonzero(available), size=max_studies - len(selected), replace=False,
        ).tolist())
    return ordered.iloc[sorted(selected)].reset_index(drop=True).copy(deep=True)


@contextmanager
def preserve_training_state(model: nn.Module) -> Iterator[nn.Module]:
    """Temporarily evaluate a model without advancing training RNGs or modes.

    Save Python, NumPy, PyTorch and CUDA random states and each module's training
    flag. Evaluation mode disables dropout and batch-normalization updates.
    The caller should also use ``torch.inference_mode()`` (as ``predict_loader``
    does) to avoid building a gradient graph. Both RNGs and mixed child modes are
    restored even if evaluation raises. Parameters and buffers are not cloned:
    this context does not undo explicit parameter or buffer edits by its caller.
    """
    random_state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield model
    finally:
        try:
            # Module.train() recurses into its children and can erase previously
            # mixed modes; assigning each original flag preserves them exactly.
            for module, training in modes:
                module.training = training
        finally:
            restore_rng(random_state)
