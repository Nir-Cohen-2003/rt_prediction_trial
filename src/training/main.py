import argparse
from pathlib import Path
import yaml
from dataclasses import asdict

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.training.trainer import train_from_config


def load_yaml_to_dataclass(path: Path | None, cls):
    """Load YAML file and instantiate dataclass."""
    if path is None:
        return cls()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    return cls(**data)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RT prediction model")
    
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rt_baseline",
        help="Name of the experiment"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load configurations from YAML files
    print(f"Loading data config from: {args.data_config}")
    data_cfg = load_yaml_to_dataclass(args.data_config, DataConfig)
    
    print(f"Loading model config from: {args.model_config}")
    model_cfg = load_yaml_to_dataclass(args.model_config, ModelConfig)
    
    print(f"Loading training config from: {args.training_config}")
    training_cfg = load_yaml_to_dataclass(args.training_config, TrainingConfig)
    
    # Create main config
    config = Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        experiment_name=args.experiment_name
    )
    
    print(f"\nStarting experiment: {config.experiment_name}")
    print(f"Data path: {config.data.raw_data_path}")
    print(f"Model type: {config.model.model_type}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}\n")
    
    # Train the model
    trainer, module, datamodule = train_from_config(config)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()