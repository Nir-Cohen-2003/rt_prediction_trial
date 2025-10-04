import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from typing import Optional
import json
from dataclasses import asdict

from chemprop.models import MPNN
from chemprop.nn import RegressionFFN, BondMessagePassing

from ..config import Config, DataConfig, ModelConfig, TrainingConfig
from ..data.datamodule import RTDataModule


class ChempropRTModule(L.LightningModule):
    """
    Lightning module wrapping Chemprop for RT prediction.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        rt_mean: float,
        rt_std: float
    ):
        """
        Args:
            model_config: Model configuration
            training_config: Training configuration
            rt_mean: Mean RT for denormalization
            rt_std: Std RT for denormalization
        """
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.rt_mean = rt_mean
        self.rt_std = rt_std
        
        # Build Chemprop model
        self.model = self._build_chemprop_model()
        
        # Loss function (MSE for regression)
        self.loss_fn = torch.nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])
    
    def _build_chemprop_model(self) -> MPNN:
        """Build Chemprop MPNN model."""
        cfg = self.model_config
        
        # Message passing
        message_passing = BondMessagePassing(
            d_h=cfg.message_hidden_dim,
            depth=cfg.num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            # aggregation=cfg.aggregation
        )
        
        # Feed-forward network
        ffn = RegressionFFN(
            input_dim=cfg.message_hidden_dim,
            hidden_dim=cfg.ffn_hidden_dim,
            n_layers=cfg.ffn_num_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            n_tasks=1  # Single RT value
        )
        
        # Full model
        model = MPNN(
            message_passing=message_passing,
            agg=None,  # Aggregation handled in message passing
            predictor=ffn,
            batch_norm=False,
            metrics=None
        )
        
        return model
    
    def forward(self, batch):
        """Forward pass."""
        return self.model(batch)
    
    def _shared_step(self, batch, batch_idx: int, stage: str):
        """Shared step for train/val/test."""
        # Forward pass
        preds = self(batch).squeeze(-1)
        targets = batch.y.squeeze(-1)
        
        # Compute loss (on normalized values)
        loss = self.loss_fn(preds, targets)
        
        # Denormalize for metrics
        preds_denorm = preds * self.rt_std + self.rt_mean
        targets_denorm = targets * self.rt_std + self.rt_mean
        
        # Compute metrics
        mae = torch.abs(preds_denorm - targets_denorm).mean()
        rmse = torch.sqrt(torch.pow(preds_denorm - targets_denorm, 2).mean())
        
        # Log
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, prog_bar=(stage != "train"), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/rmse", rmse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
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
        """Configure optimizer and scheduler."""
        cfg = self.training_config
        
        # Optimizer
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
        
        # Scheduler (optional)
        if not cfg.use_scheduler:
            return optimizer
        
        if cfg.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.monitor_mode,
                patience=cfg.scheduler_patience,
                factor=cfg.scheduler_factor,
                verbose=True
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


def train_from_config(config: Config) -> tuple[L.Trainer, ChempropRTModule, RTDataModule]:
    """
    Main training function.
    
    Args:
        config: Complete configuration object
    
    Returns:
        trainer: Lightning Trainer
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
    
    # Prepare and setup data
    datamodule.prepare_data()
    datamodule.setup()
    
    # Initialize model
    print("[train_from_config] Initializing model...")
    if config.model.model_type == "chemprop":
        module = ChempropRTModule(
            model_config=config.model,
            training_config=config.training,
            rt_mean=datamodule.rt_mean,
            rt_std=datamodule.rt_std
        )
    else:
        raise NotImplementedError("Custom GNN not implemented yet")
    
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
    
    # Save configuration
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
    
    # Train
    print("[train_from_config] Starting training...")
    trainer.fit(module, datamodule=datamodule)
    
    # Test with best checkpoint
    print("[train_from_config] Testing with best checkpoint...")
    if checkpoint_callback.best_model_path:
        trainer.test(module, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
    else:
        trainer.test(module, datamodule=datamodule)
    
    print("[train_from_config] Training complete!")
    return trainer, module, datamodule


