import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import json
import argparse
import yaml
from ..config import Config, DataConfig, ModelConfig, TrainingConfig

from ..data.datamodule import RTDataModule
from ..model.model import build_model

torch.set_float32_matmul_precision('medium')
class ChempropRTModule(L.LightningModule):
    """
    Lightning module wrapping models for RT prediction.
    
    This module handles:
    - Forward pass (supports both Chemprop and PyG models)
    - Training/validation/test steps with loss computation
    - Optimizer and scheduler configuration
    - Metric logging (MAE, RMSE on denormalized values)
    """
    
    def __init__(
        self,
        model,
        model_type: str,
        training_config: TrainingConfig,
        rt_mean: float,
        rt_std: float
    ):
        """
        Args:
            model: Pre-built model (Chemprop MPNN or PyG GNN)
            model_type: "chemprop" or "pyg"
            training_config: Training configuration (optimizer, scheduler, etc.)
            rt_mean: Mean RT for denormalization
            rt_std: Std RT for denormalization
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.training_config = training_config
        self.rt_mean = rt_mean
        self.rt_std = rt_std
        
        # Loss function (MSE for regression)
        self.loss_fn = torch.nn.MSELoss()
        
        # Save hyperparameters (excluding the model itself to avoid duplication)
        self.save_hyperparameters(ignore=["model"])
    
    def forward(self, batch):
        """
        Forward pass through the model.
        
        Args:
            batch: Chemprop batch (with .bmg attribute) or PyG batch (Data object)
        
        Returns:
            Predictions (normalized RT values)
        """
        if self.model_type == "chemprop":
            # Extract the BatchMolGraph from the TrainingBatch
            return self.model(batch.bmg)
        else:
            # PyG models expect the batch directly
            return self.model(batch)
    
    def _shared_step(self, batch, batch_idx: int, stage: str):
        """
        Shared step logic for train/val/test.
        
        Args:
            batch: Chemprop batch or PyG batch
            batch_idx: Batch index
            stage: One of "train", "val", or "test"
        
        Returns:
            Loss value
        """
        # Forward pass
        preds = self(batch).squeeze(-1)
        
        # Extract targets depending on model type
        if self.model_type == "chemprop":
            targets = batch.Y.squeeze(-1)
        else:
            # PyG batch
            targets = batch.y.squeeze(-1)
        
        # Get actual batch size
        batch_size = len(targets)
        
        # Compute loss (on normalized values)
        loss = self.loss_fn(preds, targets)
        
        # Denormalize for interpretable metrics
        preds_denorm = preds * self.rt_std + self.rt_mean
        targets_denorm = targets * self.rt_std + self.rt_mean
        
        # Compute metrics on denormalized values
        mae = torch.abs(preds_denorm - targets_denorm).mean()
        rmse = torch.sqrt(torch.pow(preds_denorm - targets_denorm, 2).mean())
        
        # Log metrics with explicit batch_size
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            f"{stage}/mae",
            mae,
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        self.log(
            f"{stage}/rmse",
            rmse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        
        return loss
    
    def training_step(self, batch, batch_idx: int):
        """Training step."""
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx: int):
        """Validation step."""
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx: int):
        """Test step."""
        return self._shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """
        Configure optimizer and optional learning rate scheduler.
        
        Returns:
            Optimizer or dict with optimizer and scheduler
        """
        cfg = self.training_config
        
        # Select and initialize optimizer
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
        
        # Return early if no scheduler
        if not cfg.use_scheduler:
            return optimizer
        
        # Configure learning rate scheduler
        if cfg.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.monitor_mode,
                patience=cfg.scheduler_patience,
                factor=cfg.scheduler_factor
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": cfg.monitor_metric
                }
            }
        elif cfg.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.num_epochs
            )
        elif cfg.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.scheduler_patience,
                gamma=cfg.scheduler_factor
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")
        
        return [optimizer], [scheduler]


def train_from_config(config: Config) -> tuple[L.Trainer, L.LightningModule, RTDataModule]:
    """
    Main training function that orchestrates the entire pipeline.
    
    This function:
    1. Sets random seeds for reproducibility
    2. Initializes the data module and prepares data
    3. Builds the model from configuration
    4. Wraps the model in a Lightning module
    5. Sets up callbacks (checkpointing, early stopping, LR monitoring)
    6. Configures logging
    7. Trains the model
    8. Evaluates on test set
    
    Args:
        config: Complete configuration object containing data, model, and training configs
    
    Returns:
        trainer: Lightning Trainer instance
        module: Trained Lightning module
        datamodule: DataModule with processed data
    """
    print("[train_from_config] Starting training pipeline")
    
    # Set seeds for reproducibility
    L.seed_everything(config.training.seed, workers=True)
    
    # Initialize datamodule
    print("[train_from_config] Initializing datamodule...")
    datamodule = RTDataModule(
        config=config.data,
        model_type=config.model.model_type,
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    # Prepare and setup data (loads or processes data)
    datamodule.prepare_data()
    datamodule.setup()
    
    # Build model using generic factory
    print(f"[train_from_config] Building {config.model.model_type} model...")
    model = build_model(config.model)
    
    # Wrap in Lightning module
    module = ChempropRTModule(
        model=model,
        model_type=config.model.model_type,
        training_config=config.training,
        rt_mean=datamodule.rt_mean,
        rt_std=datamodule.rt_std
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoint_dir / config.experiment_name,
        filename="best_model",
        monitor=config.training.monitor_metric,
        mode=config.training.monitor_mode,
        save_top_k=config.training.save_top_k,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.early_stop_patience,
        mode=config.training.monitor_mode,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Setup logger
    logger = CSVLogger(
        save_dir=config.training.log_dir,
        name=config.experiment_name
    )
    
    # Save configuration to log directory for reproducibility
    config_path = Path(logger.log_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"[train_from_config] Saved config to {config_path}")
    
    # Initialize trainer
    print("[train_from_config] Initializing trainer...")
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=config.training.deterministic,
        enable_progress_bar=True
    )
    
    # Train the module
    print("[train_from_config] Starting training...")
    trainer.fit(module, datamodule=datamodule)
    
    # Test with best checkpoint
    print("[train_from_config] Testing with best checkpoint...")
    if checkpoint_callback.best_model_path:
        print(f"[train_from_config] Loading best model from {checkpoint_callback.best_model_path}")
        trainer.test(module, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    else:
        print("[train_from_config] No best checkpoint found, testing with final model")
        trainer.test(module, datamodule=datamodule)
    
    print("[train_from_config] Training complete!")
    return trainer, module, datamodule



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
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the experiment"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for the experiment (space-separated)"
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
        experiment_name=args.experiment_name,
        description=args.description,
        tags=args.tags
    )
    
    print(f"\nStarting experiment: {config.experiment_name}")
    if config.description:
        print(f"Description: {config.description}")
    if config.tags:
        print(f"Tags: {', '.join(config.tags)}")
    print(f"Data path: {config.data.raw_data_path}")
    print(f"Model type: {config.model.model_type}")
    
    # Print CheMeleon-specific info
    if config.model.model_type == "chemprop" and config.model.chemprop.use_chemeleon:
        print(f"Using CheMeleon: {config.model.chemprop.chemeleon_checkpoint}")
        print(f"Freeze encoder: {config.model.chemprop.freeze_chemeleon}")
    
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}\n")
    
    # Train the model
    trainer, module, datamodule = train_from_config(config)
    
    print("\nTraining completed successfully!")
    print(f"Logs saved to: {config.training.log_dir}/{config.experiment_name}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}/{config.experiment_name}")


if __name__ == "__main__":
    main()