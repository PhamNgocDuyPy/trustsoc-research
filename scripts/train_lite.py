from src.config import get_config
from src.data_loader import load_processed_split
from src.models.trustsoc_lite import train_lite


if __name__ == "__main__":
    config = get_config()
    train_lite(
        load_processed_split(config, "train"),
        load_processed_split(config, "val"),
        load_processed_split(config, "test"),
        config,
        model_name="lite",
    )
