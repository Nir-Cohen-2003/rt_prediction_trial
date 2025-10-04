import argparse
import json
from pathlib import Path
import optuna
import torch
import lightning as L
import yaml
import gc
from ..config import DataConfig, ModelConfig, TrainingConfig
from .trainer import ChempropRTModule
from ..data.datamodule import RTDataModule
from ..model.chemprop_model import build_chemprop_mpnn


def load_yaml_to_dataclass(path: Path | None, cls):
    """Load YAML file into a dataclass instance."""
    if path is None:
        # Try to create with defaults, but this may fail for required fields
        try:
            return cls()
        except TypeError:
            raise ValueError(f"Cannot create {cls.__name__} without configuration file")
    
    data = yaml.safe_load(Path(path).read_text())
    
    # Try to create instance directly from dict (works if cls supports **kwargs)
    try:
        return cls(**data)
    except TypeError:
        # Fallback: create empty instance and set attributes
        inst = cls.__new__(cls)
        for k, v in data.items():
            setattr(inst, k, v)
        return inst


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Hyperparameter tuning for RT prediction (minimize val MAE)")
    p.add_argument("--data-config", type=Path, required=True, help="Path to data configuration YAML")
    p.add_argument("--tuning-config", type=Path, required=True, help="Path to tuning configuration YAML")
    return p.parse_args()


def build_objective(data_cfg: DataConfig,
                    tuning_cfg: dict):
    """
    Build the objective function for Optuna optimization.
    
    Args:
        data_cfg: Data configuration (constant across trials)
        tuning_cfg: Dictionary with tuning parameters
    
    Returns:
        objective: Function that takes a trial and returns metric to optimize
    """
    
    def objective(trial: optuna.Trial):
        try:
            # Sample hyperparameters from search space
            
            # Model hyperparameters
            message_hidden_dim = trial.suggest_categorical(
                "message_hidden_dim",
                tuning_cfg.get("message_hidden_dim_choices", [300])
            )
            num_layers = trial.suggest_int(
                "num_layers",
                tuning_cfg.get("num_layers_min", 2),
                tuning_cfg.get("num_layers_max", 5)
            )
            ffn_hidden_dim = trial.suggest_categorical(
                "ffn_hidden_dim",
                tuning_cfg.get("ffn_hidden_dim_choices", [300])
            )
            ffn_num_layers = trial.suggest_int(
                "ffn_num_layers",
                tuning_cfg.get("ffn_num_layers_min", 1),
                tuning_cfg.get("ffn_num_layers_max", 3)
            )
            dropout = trial.suggest_float(
                "dropout",
                tuning_cfg.get("dropout_min", 0.0),
                tuning_cfg.get("dropout_max", 0.5)
            )
            activation = trial.suggest_categorical(
                "activation",
                tuning_cfg.get("activation_choices", ["relu"])
            )
            aggregation = trial.suggest_categorical(
                "aggregation",
                tuning_cfg.get("aggregation_choices", ["mean"])
            )
            
            # Training hyperparameters
            learning_rate = trial.suggest_float(
                "learning_rate",
                tuning_cfg.get("lr_min", 1e-5),
                tuning_cfg.get("lr_max", 1e-2),
                log=True
            )
            batch_size = trial.suggest_categorical(
                "batch_size",
                tuning_cfg.get("batch_size_choices", [64])
            )
            optimizer = trial.suggest_categorical(
                "optimizer",
                tuning_cfg.get("optimizer_choices", ["adam"])
            )
            
            # Fix: Use log=True only if weight_decay_min > 0
            weight_decay_min = tuning_cfg.get("weight_decay_min", 0.0)
            weight_decay_max = tuning_cfg.get("weight_decay_max", 0.01)
            
            if weight_decay_min <= 0:
                # Don't use log scale if min is 0
                weight_decay = trial.suggest_float(
                    "weight_decay",
                    weight_decay_min,
                    weight_decay_max,
                    log=False
                )
            else:
                weight_decay = trial.suggest_float(
                    "weight_decay",
                    weight_decay_min,
                    weight_decay_max,
                    log=True
                )
            
            # Optional scheduler parameters
            use_scheduler = tuning_cfg.get("tune_scheduler", False)
            scheduler_type = None
            scheduler_patience = None
            scheduler_factor = None
            
            if use_scheduler:
                scheduler_type = trial.suggest_categorical(
                    "scheduler_type",
                    tuning_cfg.get("scheduler_type_choices", ["plateau"])
                )
                if scheduler_type == "plateau":
                    scheduler_patience = trial.suggest_int(
                        "scheduler_patience",
                        tuning_cfg.get("scheduler_patience_min", 5),
                        tuning_cfg.get("scheduler_patience_max", 20)
                    )
                    scheduler_factor = trial.suggest_float(
                        "scheduler_factor",
                        tuning_cfg.get("scheduler_factor_min", 0.1),
                        tuning_cfg.get("scheduler_factor_max", 0.7)
                    )
            
            # Create model configuration with sampled hyperparameters
            model_cfg = ModelConfig(
                model_type="chemprop",  # Constant for now
                message_hidden_dim=message_hidden_dim,
                num_layers=num_layers,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_num_layers=ffn_num_layers,
                dropout=dropout,
                activation=activation,
                aggregation=aggregation,
                use_additional_features=data_cfg.additional_features is not None and len(data_cfg.additional_features) > 0,
                additional_feature_dim=len(data_cfg.additional_features) if data_cfg.additional_features else 0
            )
            
            # Create training configuration with sampled hyperparameters
            training_cfg = TrainingConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=tuning_cfg.get("epochs_per_trial", 50),
                optimizer=optimizer,
                weight_decay=weight_decay,
                use_scheduler=use_scheduler,
                scheduler_type=scheduler_type if use_scheduler else "plateau",
                scheduler_patience=scheduler_patience if use_scheduler else 10,
                scheduler_factor=scheduler_factor if use_scheduler else 0.5,
                early_stop_patience=tuning_cfg.get("early_stop_patience", 15),
                monitor_metric="val/mae",
                monitor_mode="min"
            )
            
            # Set seed for reproducibility (unique per trial)
            if tuning_cfg.get("seed") is not None:
                L.seed_everything(tuning_cfg["seed"] + trial.number, workers=True)
            
            # Create new datamodule for this trial with the new batch size
            dm = RTDataModule(
                config=data_cfg,
                model_type=model_cfg.model_type,
                batch_size=batch_size,
                num_workers=4
            )
            # Setup will load pre-processed data (fast)
            dm.setup()
            
            # Build model
            if model_cfg.model_type == "chemprop":
                model = build_chemprop_mpnn(model_cfg)
                module = ChempropRTModule(
                    model=model,
                    training_config=training_cfg,
                    rt_mean=dm.rt_mean,
                    rt_std=dm.rt_std
                )
            else:
                raise NotImplementedError(f"Model type '{model_cfg.model_type}' not implemented")
            
            # Create minimal trainer (no checkpointing or logging for speed)
            trainer = L.Trainer(
                max_epochs=training_cfg.num_epochs,
                accelerator=training_cfg.accelerator,
                devices=training_cfg.devices,
                precision=training_cfg.precision,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                callbacks=[]
            )
            
            # Train the model
            trainer.fit(module, datamodule=dm)
            
            # Get validation metrics
            try:
                val_results = trainer.validate(module, datamodule=dm, verbose=False)
                val_metrics = val_results[0] if isinstance(val_results, (list, tuple)) and len(val_results) > 0 else {}
            except Exception:
                val_metrics = {}
            
            # Extract the metric we want to optimize (MAE)
            metric = None
            for k in ("val/mae", "val_mae", "mae"):
                if k in val_metrics:
                    metric = val_metrics[k]
                    break
            
            if metric is None:
                metric = trainer.callback_metrics.get("val/mae") or \
                         trainer.callback_metrics.get("val_mae") or \
                         trainer.callback_metrics.get("mae")
            
            if metric is None:
                return float("inf")  # Return high value if metric not found (we're minimizing)
            
            return float(metric.item()) if isinstance(metric, torch.Tensor) else float(metric)
        
        finally:
            # Clean up to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return objective


def main():
    """Main entry point for hyperparameter tuning."""
    args = parse_args()
    
    # Load data configuration (constant across all trials)
    print("[hyperparam_tune] Loading data configuration...")
    data_cfg = load_yaml_to_dataclass(args.data_config, DataConfig)
    
    # Load tuning config as dictionary
    print("[hyperparam_tune] Loading tuning configuration...")
    tuning_cfg = yaml.safe_load(Path(args.tuning_config).read_text())
    
    # Override data config with tuning config if specified
    if "data_dir" in tuning_cfg:
        data_cfg.data_dir = tuning_cfg["data_dir"]
    if "train_file" in tuning_cfg:
        data_cfg.train_file = tuning_cfg["train_file"]
    if "val_file" in tuning_cfg:
        data_cfg.val_file = tuning_cfg["val_file"]
    if "test_file" in tuning_cfg:
        data_cfg.test_file = tuning_cfg["test_file"]
    if "smiles_column" in tuning_cfg:
        data_cfg.smiles_column = tuning_cfg["smiles_column"]
    if "target_column" in tuning_cfg:
        data_cfg.target_column = tuning_cfg["target_column"]
    if "additional_features" in tuning_cfg:
        data_cfg.additional_features = tuning_cfg["additional_features"]
    
    print(f"[hyperparam_tune] Data config: {data_cfg}")
    
    # Prepare data once (this creates the processed data if it doesn't exist)
    print("[hyperparam_tune] Preparing data (one-time processing)...")
    dm_temp = RTDataModule(
        config=data_cfg,
        model_type="chemprop",
        batch_size=64,
        num_workers=4
    )
    dm_temp.prepare_data()  # This processes and saves the data
    print("[hyperparam_tune] Data preparation complete. Trials will reuse processed data.")
    
    # Set global seed
    if tuning_cfg.get("seed") is not None:
        L.seed_everything(tuning_cfg["seed"], workers=True)
    
    # Create Optuna study
    print("[hyperparam_tune] Creating Optuna study...")
    study = optuna.create_study(
        study_name=tuning_cfg.get("study_name", "rt_prediction_tuning"),
        direction=tuning_cfg.get("direction", "minimize"),  # Minimize MAE
        storage=tuning_cfg.get("storage"),
        load_if_exists=bool(tuning_cfg.get("storage"))
    )
    
    # Build objective function
    obj = build_objective(data_cfg, tuning_cfg)
    
    # Run optimization with parallel trials support
    n_jobs = tuning_cfg.get("n_jobs", 1)
    print(f"[hyperparam_tune] Starting optimization with {tuning_cfg.get('trials', 100)} trials...")
    print(f"[hyperparam_tune] Running {n_jobs} trial(s) in parallel...")
    
    study.optimize(
        obj,
        n_trials=tuning_cfg.get("trials", 100),
        n_jobs=n_jobs
    )
    
    # Print and save results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val MAE: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    output = {
        "best_trial": study.best_trial.number,
        "best_val_mae": study.best_value,
        "best_params": study.best_trial.params,
        "study_name": tuning_cfg.get("study_name", "rt_prediction_tuning"),
        "num_trials": len(study.trials)
    }
    
    out_path = Path(tuning_cfg.get("output_path", "results/tuning_results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[hyperparam_tune] Results saved to {out_path}")


if __name__ == "__main__":
    main()