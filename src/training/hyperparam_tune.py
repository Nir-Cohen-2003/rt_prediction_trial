import argparse
import json
from pathlib import Path
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import torch
import lightning as L
import yaml
import gc
import os
from datetime import datetime
from ..config import DataConfig, ModelConfig, TrainingConfig
from .trainer import ChempropRTModule
from ..data.datamodule import RTDataModule
from ..model.chemprop_model import build_chemprop_mpnn


def load_yaml_to_dataclass(path: Path | None, cls):
    """Load YAML file into a dataclass instance."""
    if path is None:
        raise ValueError(f"Cannot create {cls.__name__} without configuration file")
    
    data = yaml.safe_load(Path(path).read_text())
    
    # Try to create instance directly from dict (works if cls supports **kwargs)
    try:
        return cls(**data)
    except TypeError as e:
        raise ValueError(f"Failed to create {cls.__name__} from config file: {e}")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Hyperparameter tuning for RT prediction (minimize val MAE)")
    p.add_argument("--data-config", type=Path, required=True, help="Path to data configuration YAML")
    p.add_argument("--tuning-config", type=Path, required=True, help="Path to tuning configuration YAML")
    return p.parse_args()


def save_trial_result(trial: optuna.Trial, value: float, output_dir: Path):
    """Save individual trial result immediately after completion."""
    trial_file = output_dir / f"trial_{trial.number:04d}.json"
    
    trial_data = {
        "trial_number": trial.number,
        "value": value,
        "params": trial.params,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None,
        "state": trial.state.name,
    }
    
    trial_file.write_text(json.dumps(trial_data, indent=2))


def save_current_best(study: optuna.Study, output_dir: Path):
    """Save current best results in both JSON and human-readable format."""
    
    # Machine-readable JSON
    best_json = output_dir / "best_result.json"
    best_data = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "num_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "num_total_trials": len(study.trials),
        "last_updated": datetime.now().isoformat(),
    }
    best_json.write_text(json.dumps(best_data, indent=2))
    
    # Human-readable text
    best_txt = output_dir / "best_result.txt"
    lines = [
        "=" * 80,
        "CURRENT BEST HYPERPARAMETERS",
        "=" * 80,
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} / {len(study.trials)}",
        "",
        f"Best Trial Number: {study.best_trial.number}",
        f"Best Validation MAE: {study.best_value:.6f}",
        "",
        "Best Hyperparameters:",
        "-" * 80,
    ]
    
    # Group parameters by category
    model_params = {}
    training_params = {}
    scheduler_params = {}
    
    for key, value in study.best_trial.params.items():
        if key in ["message_hidden_dim", "num_layers", "ffn_hidden_dim", "ffn_num_layers", 
                   "dropout", "activation", "aggregation"]:
            model_params[key] = value
        elif key.startswith("scheduler_"):
            scheduler_params[key] = value
        else:
            training_params[key] = value
    
    if model_params:
        lines.append("\n[Model Architecture]")
        for key, value in sorted(model_params.items()):
            lines.append(f"  {key:25s}: {value}")
    
    if training_params:
        lines.append("\n[Training]")
        for key, value in sorted(training_params.items()):
            if isinstance(value, float) and value < 0.01:
                lines.append(f"  {key:25s}: {value:.6f}")
            else:
                lines.append(f"  {key:25s}: {value}")
    
    if scheduler_params:
        lines.append("\n[Learning Rate Scheduler]")
        for key, value in sorted(scheduler_params.items()):
            lines.append(f"  {key:25s}: {value}")
    
    lines.append("\n" + "=" * 80)
    
    best_txt.write_text("\n".join(lines))


def save_all_trials_summary(study: optuna.Study, output_dir: Path):
    """Save summary of all trials in human-readable format."""
    summary_file = output_dir / "trials_summary.txt"
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    lines = [
        "=" * 80,
        "HYPERPARAMETER TUNING SUMMARY",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total trials: {len(study.trials)}",
        f"Completed: {len(completed_trials)}",
        f"Failed: {len(failed_trials)}",
        "",
        f"Best trial: #{study.best_trial.number}",
        f"Best validation MAE: {study.best_value:.6f}",
        "",
        "=" * 80,
        "TOP 10 TRIALS",
        "=" * 80,
        "",
    ]
    
    # Sort completed trials by value
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
    
    for rank, trial in enumerate(sorted_trials, 1):
        lines.append(f"Rank {rank}: Trial #{trial.number}")
        lines.append(f"  Validation MAE: {trial.value:.6f}")
        lines.append(f"  Key params:")
        for key in ["message_hidden_dim", "num_layers", "learning_rate", "batch_size"]:
            if key in trial.params:
                value = trial.params[key]
                if isinstance(value, float) and value < 0.01:
                    lines.append(f"    {key}: {value:.6f}")
                else:
                    lines.append(f"    {key}: {value}")
        lines.append("")
    
    lines.append("=" * 80)
    
    summary_file.write_text("\n".join(lines))


def build_objective(data_cfg: DataConfig,
                    tuning_cfg: dict,
                    output_dir: Path):
    """
    Build the objective function for Optuna optimization.
    
    Args:
        data_cfg: Data configuration (constant across trials)
        tuning_cfg: Dictionary with tuning parameters
        output_dir: Directory to save results
    
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
                model_type="chemprop",
                message_hidden_dim=message_hidden_dim,
                num_layers=num_layers,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_num_layers=ffn_num_layers,
                dropout=dropout,
                activation=activation,
                aggregation=aggregation,
                use_additional_features=False,
                additional_feature_dim=0
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
            
            # GPU allocation: Round-robin assignment across available GPUs
            n_jobs = tuning_cfg.get("n_jobs", 1)
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                # Assign GPU based on trial number (allows multiple trials per GPU)
                gpu_id = trial.number % n_gpus
                
                # Set CUDA_VISIBLE_DEVICES for this trial to prevent cross-GPU interference
                # This isolates the trial to a specific GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                # Use GPU with single device
                accelerator = "gpu"
                devices = 1  # Use first device (which is the GPU we set in CUDA_VISIBLE_DEVICES)
                
                # Reduce num_workers for parallel trials to avoid dataloader worker overhead
                num_workers = 0 if n_jobs > 1 else 2
            else:
                raise RuntimeError("No GPU available! This script requires GPU.")
            
            # Create new datamodule for this trial with the new batch size
            dm = RTDataModule(
                config=data_cfg,
                model_type=model_cfg.model_type,
                batch_size=batch_size,
                num_workers=num_workers
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
                accelerator=accelerator,
                devices=devices,
                precision=training_cfg.precision,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                callbacks=[],
                # Enable GPU memory sharing optimizations
                benchmark=True,  # cudnn.benchmark for faster training
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
            
            value = float(metric.item()) if isinstance(metric, torch.Tensor) else float(metric)
            
            # Save this trial's result immediately
            save_trial_result(trial, value, output_dir)
            
            return value
        
        finally:
            # Clean up to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return objective


def main():
    """Main entry point for hyperparameter tuning."""
    args = parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available! This script requires GPU for training.")
    
    n_gpus = torch.cuda.device_count()
    print(f"[hyperparam_tune] Found {n_gpus} GPU(s)")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load data configuration (constant across all trials)
    print("[hyperparam_tune] Loading data configuration...")
    data_cfg = load_yaml_to_dataclass(args.data_config, DataConfig)
    
    # Load tuning config as dictionary
    print("[hyperparam_tune] Loading tuning configuration...")
    tuning_cfg = yaml.safe_load(Path(args.tuning_config).read_text())
    
    print(f"[hyperparam_tune] Data config: {data_cfg}")
    
    # Create output directory
    output_dir = Path(tuning_cfg.get("output_path", "results/tuning_results")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for this run with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = tuning_cfg.get("study_name", "rt_prediction_tuning")
    run_dir = output_dir / f"{study_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[hyperparam_tune] Results will be saved to: {run_dir}")
    
    # Save the tuning configuration for reference
    config_copy = run_dir / "tuning_config.yaml"
    config_copy.write_text(yaml.dump(tuning_cfg, default_flow_style=False, sort_keys=False))
    
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
    
    # Create persistent storage using SQLite for crash recovery
    storage_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"
    
    # Create Optuna study with persistent storage
    print("[hyperparam_tune] Creating Optuna study...")
    study = optuna.create_study(
        study_name=study_name,
        direction=tuning_cfg.get("direction", "minimize"),  # Minimize MAE
        storage=storage,
        load_if_exists=True  # Resume if crashed
    )
    
    print(f"[hyperparam_tune] Study storage: {storage_path}")
    
    # Build objective function
    obj = build_objective(data_cfg, tuning_cfg, run_dir)
    
    # Add callback to save best result after each trial
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            save_current_best(study, run_dir)
            save_all_trials_summary(study, run_dir)
            print(f"[Trial {trial.number}] Value: {trial.value:.6f} | Best so far: {study.best_value:.6f}")
    
    # Run optimization with parallel trials support
    n_jobs = tuning_cfg.get("n_jobs", 1)
    n_trials = tuning_cfg.get("trials", 100)
    
    print(f"\n[hyperparam_tune] Starting optimization:")
    print(f"  Total trials: {n_trials}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Available GPUs: {n_gpus}")
    print(f"  Trials per GPU: ~{n_jobs / n_gpus:.1f}")
    print()
    
    study.optimize(
        obj,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[callback]
    )
    
    # Final summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val MAE: {study.best_value:.6f}")
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - Individual trials: trial_XXXX.json")
    print(f"  - Current best: best_result.json and best_result.txt")
    print(f"  - Summary: trials_summary.txt")
    print(f"  - Study database: optuna_study.db (for resuming)")


if __name__ == "__main__":
    main()