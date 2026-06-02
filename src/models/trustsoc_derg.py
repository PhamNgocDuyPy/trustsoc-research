from __future__ import annotations

from typing import Any

import pandas as pd

from dataclasses import replace

from ..config import ProjectConfig
from .trustsoc_lite import train_lite


def train_derg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    model_name: str = "trustsoc_derg",
) -> dict[str, Any]:
    derg_config = replace(config, text_max_word_features=min(config.text_max_word_features, 20000), text_max_char_features=min(config.text_max_char_features, 12000))
    return train_lite(
        train_df,
        val_df,
        test_df,
        derg_config,
        model_name=model_name,
        include_text=True,
        include_cti=True,
        include_mitre=True,
        include_derg=True,
        include_adversarial=True,
    )
