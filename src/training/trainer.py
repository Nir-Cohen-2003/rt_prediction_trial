import torch
import torch.nn as nn
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
import math
torch.set_float32_matmul_precision('medium')

class GradientClippingCallback(L.Callback):
    """Monitor and log gradient norms to detect instabilities."""
    
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Log gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), 
            max_norm=float('inf')  # Just compute, don't clip yet
        )
        pl_module.log('train/grad_norm', grad_norm, prog_bar=False)

class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_epochs
    2. Cosine annealing from base_lr to eta_min over remaining epochs
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of epochs
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


class RTTrainer(L.LightningModule):
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
        
        # Select loss function
        if training_config.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif training_config.loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        elif training_config.loss_fn == "huber":
            self.loss_fn = nn.HuberLoss(delta=training_config.huber_delta)
        elif training_config.loss_fn == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(beta=training_config.huber_delta)
        else:
            raise ValueError(f"Unknown loss function: {training_config.loss_fn}")
        
        print(f"[ChempropRTModule] Using loss function: {training_config.loss_fn}")
        if training_config.loss_fn in ["huber", "smooth_l1"]:
            print(f"[ChempropRTModule] Huber delta: {training_config.huber_delta}")
        
        # Track best validation loss for anomaly detection
        self.best_val_loss = float('inf')
        self.val_loss_spike_count = 0
        
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
        
        # Check for NaN/Inf in loss during training
        if stage == "train" and (torch.isnan(loss) or torch.isinf(loss)):
            self.log('train/nan_loss', 1.0, batch_size=batch_size)
            # Return a safe loss to prevent crash
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
        
        # Denormalize for interpretable metrics
        preds_denorm = preds * self.rt_std + self.rt_mean
        targets_denorm = targets * self.rt_std + self.rt_mean
        
        # Compute metrics on denormalized values
        mae = torch.abs(preds_denorm - targets_denorm).mean()
        rmse = torch.sqrt(torch.pow(preds_denorm - targets_denorm, 2).mean())
        # R2 score
        ss_res = torch.sum((preds_denorm - targets_denorm) ** 2)
        ss_tot = torch.sum((targets_denorm - torch.mean(targets_denorm)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
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
        self.log(
            f"{stage}/r2",
            r2,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        
        return loss

    def on_validation_epoch_end(self):
        """Detect sudden spikes in validation loss and print metrics."""
        val_loss = self.trainer.callback_metrics.get('val/loss')
        val_mae = self.trainer.callback_metrics.get('val/mae')
        val_rmse = self.trainer.callback_metrics.get('val/rmse')
        val_r2 = self.trainer.callback_metrics.get('val/r2')
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else self.trainer.current_epoch

        # Print metrics after each validation epoch
        print(
            f"\nEpoch {current_epoch}: "
            f"val/loss={val_loss:.4f} "
            f"val/mae={val_mae:.4f} "
            f"val/rmse={val_rmse:.4f} "
            f"val/r2={val_r2:.4f}"
        )

        if val_loss is not None:
            # Detect spike (loss increased by >3x)
            if val_loss > self.best_val_loss * 3.0:
                self.val_loss_spike_count += 1
                self.log('val/loss_spike', 1.0)
                print(f"\n⚠️ WARNING: Validation loss spike detected! "
                      f"Previous best: {self.best_val_loss:.2f}, Current: {val_loss:.2f}")
            else:
                self.val_loss_spike_count = 0
            
            # Update best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
    
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
                weight_decay=cfg.weight_decay,
                eps=1e-8  # More stable epsilon
            )
        elif cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                eps=1e-8
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
                factor=cfg.scheduler_factor,
                min_lr=1e-7  # Prevent LR from going too low
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
                T_max=cfg.num_epochs,
                eta_min=1e-7  # Minimum learning rate
            )
        elif cfg.scheduler_type == "cosine_warmup":
            scheduler = CosineAnnealingWarmupScheduler(
                optimizer,
                warmup_epochs=cfg.warmup_epochs,
                max_epochs=cfg.num_epochs,
                eta_min=1e-7
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
    module = RTTrainer(
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
        verbose=True,
        check_finite=True  # Stop if loss becomes NaN/Inf
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Add gradient monitoring
    grad_monitor = GradientClippingCallback()
    
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
    
    # Fix devices parameter - convert string to appropriate type
    devices_param = config.training.devices
    if isinstance(devices_param, str):
        # Convert string like "0" or "0,1" to integer or list
        if ',' in devices_param:
            devices_param = [int(d.strip()) for d in devices_param.split(',')]
        else:
            try:
                devices_param = int(devices_param)
            except ValueError:
                devices_param = 1  # Default to 1 device
    
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator=config.training.accelerator,
        devices=devices_param,  # Use converted parameter
        precision=config.training.precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, grad_monitor],
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=config.training.deterministic,
        enable_progress_bar=False,  # Turn off progress bar
        gradient_clip_val=1.0,  # Clip gradients to prevent explosions
        gradient_clip_algorithm="norm",
        detect_anomaly=False,  # Set to True for debugging but slower
        accumulate_grad_batches=1  # Can increase for more stable gradients
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