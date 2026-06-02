from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .adversarial_generator import assign_adversarial_type, assign_expected_action, create_robustness_views
from .config import ProjectConfig
from .data_loader import load_raw_datasets
from .derg_builder import graph_for_case
from .evidence_extractor import build_cti_catalogs, build_evidence_items, serialize_json
from .utils import ensure_directories, get_logger, save_json, set_seed


def normalize_row(row: pd.Series, split_name: str, catalogs: dict[str, object]) -> dict[str, object]:
    normalized = row.to_dict()
    normalized["split"] = split_name
    normalized["raw_fields"] = normalized.get("raw_fields") or {}
    normalized["raw_fields_json"] = json.dumps(normalized["raw_fields"], ensure_ascii=False)

    evidence = build_evidence_items(normalized, catalogs)
    normalized.update(
        {
            "evidence_text": evidence["evidence_text"],
            "evidence_items": evidence["evidence_items"],
            "evidence_items_json": serialize_json(evidence["evidence_items"]),
            "cti_matches": evidence["cti_matches"],
            "cti_matches_json": serialize_json(evidence["cti_matches"]),
            "cti_match_count": evidence["cti_match_count"],
            "cti_match_score": evidence["cti_match_score"],
            "mitre_list": evidence["mitre_list"],
            "mitre_list_json": serialize_json(evidence["mitre_list"]),
            "mitre_count": evidence["mitre_count"],
            "mitre_risk_score": evidence["mitre_risk_score"],
            "rule_signals_json": serialize_json(evidence["rule_signals"]),
            "contradiction_score": evidence["contradiction_score"],
            "evidence_diversity": evidence["evidence_diversity"],
            "evidence_consistency": evidence["evidence_consistency"],
            "adversarial_noise_score": evidence["adversarial_noise_score"],
            "reliability_score": evidence["reliability_score"],
        }
    )

    graph, graph_features = graph_for_case(normalized)
    normalized["derg_features_json"] = serialize_json(graph_features)
    normalized.update(graph_features)
    normalized["adversarial_type"] = assign_adversarial_type(pd.Series(normalized))
    normalized["expected_action_target"] = assign_expected_action(pd.Series(normalized))
    normalized["explanation"] = (
        f"derived_action={normalized['expected_action_target']}; "
        f"contradiction={normalized['contradiction_score']:.2f}; "
        f"cti_matches={normalized['cti_match_count']}; "
        f"mitre_count={normalized['mitre_count']}"
    )
    normalized["evidence_node_count"] = graph.number_of_nodes()
    return normalized


def process_split(df: pd.DataFrame, split_name: str, catalogs: dict[str, object], logger) -> pd.DataFrame:
    rows = [normalize_row(row, split_name, catalogs) for _, row in df.iterrows()]
    processed = pd.DataFrame(rows)
    logger.info("Processed split %s with shape %s", split_name, processed.shape)
    return processed


def low_resource_sample(df: pd.DataFrame, train_size: int, val_size: int, test_size: int, seed: int) -> dict[str, pd.DataFrame]:
    def sample_one(frame: pd.DataFrame, size: int, offset: int) -> pd.DataFrame:
        if len(frame) <= size:
            return frame.reset_index(drop=True)
        counts = frame["threat_type"].value_counts()
        if counts.min() >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed + offset)
            idx, _ = next(splitter.split(frame, frame["threat_type"]))
            return frame.iloc[idx].sample(frac=1.0, random_state=seed + offset).reset_index(drop=True)

        weights = counts / counts.sum()
        chunks = []
        for label, weight in weights.items():
            class_rows = frame[frame["threat_type"] == label]
            take = min(len(class_rows), max(1, int(round(size * weight))))
            chunks.append(class_rows.sample(n=take, random_state=seed + offset))

        sampled = pd.concat(chunks)
        if len(sampled) > size:
            sampled = sampled.sample(n=size, random_state=seed + offset)
        elif len(sampled) < size:
            remaining = frame.drop(index=sampled.index, errors="ignore")
            if len(remaining) > 0:
                extra = remaining.sample(n=min(size - len(sampled), len(remaining)), random_state=seed + offset)
                sampled = pd.concat([sampled, extra])
        return sampled.sample(frac=1.0, random_state=seed + offset).reset_index(drop=True)

    return {
        "train": sample_one(df[df["split"] == "train"], train_size, 0),
        "val": sample_one(df[df["split"] == "val"], val_size, 1),
        "test": sample_one(df[df["split"] == "test"], test_size, 2),
    }


def write_processed_outputs(config: ProjectConfig, processed: dict[str, pd.DataFrame], low_resource: dict[str, pd.DataFrame], robustness: dict[str, pd.DataFrame]) -> None:
    for split_name, path in config.processed_split_paths.items():
        processed[split_name].to_csv(path, index=False, encoding="utf-8")
    for split_name, path in config.low_resource_split_paths.items():
        low_resource[split_name].to_csv(path, index=False, encoding="utf-8")
    for split_name, path in config.robustness_split_paths.items():
        robustness[split_name].to_csv(path, index=False, encoding="utf-8")


def preprocess_all(config: ProjectConfig | None = None) -> dict[str, object]:
    config = config or ProjectConfig()
    ensure_directories(config)
    set_seed(config.seed)
    logger = get_logger(config, "preprocess")
    raw_datasets = load_raw_datasets(config, logger)
    catalogs = build_cti_catalogs(raw_datasets)

    processed = {
        "train": process_split(raw_datasets["trustsoc_train.jsonl"], "train", catalogs, logger),
        "val": process_split(raw_datasets["trustsoc_val.jsonl"], "val", catalogs, logger),
        "test": process_split(raw_datasets["trustsoc_test.jsonl"], "test", catalogs, logger),
    }
    processed["full"] = pd.concat([processed["train"], processed["val"], processed["test"]], ignore_index=True)

    low_resource = low_resource_sample(
        processed["full"],
        config.low_resource_train_size,
        config.low_resource_val_size,
        config.low_resource_test_size,
        config.seed,
    )
    robustness = create_robustness_views(processed["test"])
    write_processed_outputs(config, processed, low_resource, robustness)

    dataset_stats = {
        "processed_rows": {name: int(df.shape[0]) for name, df in processed.items()},
        "low_resource_rows": {name: int(df.shape[0]) for name, df in low_resource.items()},
        "robustness_rows": {name: int(df.shape[0]) for name, df in robustness.items()},
    }
    save_json(config.metrics_dir / "preprocess_summary.json", dataset_stats)
    logger.info("Preprocessing complete. Summary saved to %s", config.metrics_dir / "preprocess_summary.json")
    return dataset_stats
