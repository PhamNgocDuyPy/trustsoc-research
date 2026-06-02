from __future__ import annotations

import argparse

from src.ablation import run_ablation
from src.config import get_config
from src.evaluation import compare_with_opensoc, evaluate, generate_report_tables
from src.models.sklearn_baselines import train_baselines
from src.models.trustsoc_derg import train_derg
from src.models.trustsoc_lite import train_lite
from src.models.trustsoc_transformer import train_transformer
from src.preprocessing import preprocess_all
from src.robustness import run_robustness
from src.data_loader import load_low_resource_split, load_processed_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrustSOC-Research local runner.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "preprocess",
            "train_baselines",
            "train_lite",
            "train_derg",
            "train_transformer",
            "evaluate",
            "compare_opensoc",
            "ablation",
            "robustness",
            "report",
        ],
    )
    parser.add_argument("--low-resource", action="store_true", help="Use the low-resource split for training commands.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.mode == "preprocess":
        preprocess_all(config)
        return

    if args.mode == "train_baselines":
        train_df = load_processed_split(config, "train")
        val_df = load_processed_split(config, "val")
        test_df = load_processed_split(config, "test")
        train_baselines(train_df, val_df, test_df, config)
        return

    if args.mode in {"train_lite", "train_derg", "train_transformer"}:
        if args.low_resource:
            train_df = load_low_resource_split(config, "train")
            val_df = load_low_resource_split(config, "val")
            test_df = load_low_resource_split(config, "test")
        else:
            train_df = load_processed_split(config, "train")
            val_df = load_processed_split(config, "val")
            test_df = load_processed_split(config, "test")

        if args.mode == "train_lite":
            train_lite(train_df, val_df, test_df, config, model_name="lite_low_resource" if args.low_resource else "lite")
        elif args.mode == "train_derg":
            train_derg(train_df, val_df, test_df, config, model_name="trustsoc_derg_low_resource" if args.low_resource else "trustsoc_derg")
        else:
            train_transformer(train_df, val_df, test_df, config, model_name="transformer_low_resource" if args.low_resource else "transformer")
        return

    if args.mode == "evaluate":
        evaluate(config)
        return

    if args.mode == "compare_opensoc":
        compare_with_opensoc(config)
        return

    if args.mode == "ablation":
        run_ablation(config)
        return

    if args.mode == "robustness":
        run_robustness(config)
        return

    if args.mode == "report":
        generate_report_tables(config)
        return


if __name__ == "__main__":
    main()
