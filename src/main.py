from pathlib import Path
from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.training.trainer import train_from_config

config = Config(
    data=DataConfig(
        raw_data_path=Path("data/raw/rt_data.csv"),
        output_dir=Path("data/processed"),
        split_method="random"
    ),
    model=ModelConfig(
        model_type="chemprop",
        message_hidden_dim=300,
        num_layers=3
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        batch_size=64,
        num_epochs=100
    ),
    experiment_name="rt_baseline"
)

trainer, module, datamodule = train_from_config(config)