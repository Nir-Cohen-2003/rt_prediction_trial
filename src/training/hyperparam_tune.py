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
import fcntl  # For file locking
import time
from dataclasses import asdict
from ..config import DataConfig, ModelConfig, TrainingConfig, ChemPropModelConfig,PyGModelConfig
from .trainer import RTTrainer
from ..data.datamodule import RTDataModule
from ..model.model import build_model


def load_yaml_to_dataclass(path: Path | None, cls):
    """Load YAML file into a dataclass instance."""
    if path is None:
        raise ValueError(f"Cannot create {cls.__name__} without configuration file")
    
    data = yaml.safe_load(Path(path).read_text())
    
    # Special handling for ModelConfig with nested configs
    if cls == ModelConfig and 'chemprop' in data and isinstance(data['chemprop'], dict):
        from ..config import ChemPropModelConfig
        data['chemprop'] = ChemPropModelConfig(**data['chemprop'])
    elif cls == ModelConfig and 'pyg' in data and isinstance(data['pyg'], dict):
        from ..config import PyGModelConfig
        data['pyg'] = PyGModelConfig(**data['pyg'])
    
    # Try to create instance directly from dict
    try:
        return cls(**data)
    except TypeError as e:
        raise ValueError(f"Failed to create {cls.__name__} from config file: {e}")


def initialize_pyg_dimensions(data_cfg: DataConfig, model_cfg: ModelConfig):
    """
    Initialize PyG feature dimensions from featurizer.
    This replicates the logic from Config.__post_init__.
    """
    if model_cfg.model_type != "pyg":
        return
    
    # Create appropriate featurizer to get dimensions
    if data_cfg.featurizer_type == "simple":
        featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_features=data_cfg.atom_features,
            bond_features=data_cfg.bond_features,
            atom_descriptors=data_cfg.atom_descriptors,
            bond_descriptors=data_cfg.bond_descriptors
        )
    elif data_cfg.featurizer_type == "v1":
        featurizer = MultiHotAtomFeaturizer.v1()
    elif data_cfg.featurizer_type == "v2":
        featurizer = MultiHotAtomFeaturizer.v2()
    elif data_cfg.featurizer_type == "organic":
        featurizer = MultiHotAtomFeaturizer.organic()
    elif data_cfg.featurizer_type == "rigr":
        featurizer = RIGRAtomFeaturizer()
    else:
        raise ValueError(f"Unknown featurizer_type: {data_cfg.featurizer_type}")
    
    # Set dimensions in PyG config
    model_cfg.pyg.node_in_dim = len(featurizer)
    if hasattr(featurizer, 'bond_fdim'):
        model_cfg.pyg.edge_in_dim = featurizer.bond_fdim
    
    print(f"[initialize_pyg_dimensions] Set PyG feature dimensions: "
          f"node_in_dim={model_cfg.pyg.node_in_dim}, edge_in_dim={model_cfg.pyg.edge_in_dim}")


def safe_write_json(filepath: Path, data: dict, max_retries: int = 5):
    """Thread-safe JSON file writing with file locking."""
    for attempt in range(max_retries):
        try:
            with open(filepath, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: Failed to write {filepath} after {max_retries} attempts: {e}")


def safe_write_text(filepath: Path, content: str, max_retries: int = 5):
    """Thread-safe text file writing with file locking."""
    for attempt in range(max_retries):
        try:
            with open(filepath, 'w') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: Failed to write {filepath} after {max_retries} attempts: {e}")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Hyperparameter tuning for RT prediction (minimize val MAE)")
    p.add_argument("--data-config", type=Path, required=True, help="Path to data configuration YAML")
    p.add_argument("--model-config", type=Path, required=True, help="Path to model configuration YAML")
    p.add_argument("--tuning-config", type=Path, required=True, help="Path to tuning configuration YAML")
    return p.parse_args()


def save_trial_result(trial: optuna.Trial, value: float, output_dir: Path, study: optuna.Study):
    """Save individual trial result immediately after completion."""
    trial_file = output_dir / f"trial_{trial.number:04d}.json"
    
    # Get the frozen trial from the study to access completion info
    try:
        frozen_trial = study.trials[trial.number]
        datetime_start = frozen_trial.datetime_start.isoformat() if frozen_trial.datetime_start else None
        datetime_complete = frozen_trial.datetime_complete.isoformat() if frozen_trial.datetime_complete else None
        duration_seconds = (frozen_trial.datetime_complete - frozen_trial.datetime_start).total_seconds() \
                          if frozen_trial.datetime_complete and frozen_trial.datetime_start else None
        state = frozen_trial.state.name
    except (IndexError, AttributeError):
        # Fallback if frozen trial not yet available
        datetime_start = None
        datetime_complete = None
        duration_seconds = None
        state = "RUNNING"
    
    trial_data = {
        "trial_number": trial.number,
        "value": value,
        "params": trial.params,
        "datetime_start": datetime_start,
        "datetime_complete": datetime_complete,
        "duration_seconds": duration_seconds,
        "state": state,
    }
    
    safe_write_json(trial_file, trial_data)


def save_current_best(study: optuna.Study, output_dir: Path):
    """Save current best results in both JSON and human-readable format."""
    
    # Check if there are any completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("[save_current_best] No completed trials yet, skipping save")
        return
    
    try:
        # Machine-readable JSON
        best_json = output_dir / "best_result.json"
        best_data = {
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "num_completed_trials": len(completed_trials),
            "num_total_trials": len(study.trials),
            "last_updated": datetime.now().isoformat(),
        }
        safe_write_json(best_json, best_data)
        
        # Human-readable text
        best_txt = output_dir / "best_result.txt"
        lines = [
            "=" * 80,
            "CURRENT BEST HYPERPARAMETERS",
            "=" * 80,
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Completed trials: {len(completed_trials)} / {len(study.trials)}",
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
        
        safe_write_text(best_txt, "\n".join(lines))
        print(f"[save_current_best] Saved best results to {output_dir}")
    except Exception as e:
        print(f"[save_current_best] Error saving best results: {e}")


def save_all_trials_summary(study: optuna.Study, output_dir: Path):
    """Save summary of all trials in human-readable format."""
    
    try:
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
        ]
        
        if completed_trials:
            lines.extend([
                f"Best trial: #{study.best_trial.number}",
                f"Best validation MAE: {study.best_value:.6f}",
                "",
                "=" * 80,
                "TOP 10 TRIALS",
                "=" * 80,
                "",
            ])
            
            # Sort completed trials by value
            sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:10]
            
            for rank, trial in enumerate(sorted_trials, 1):
                lines.append(f"Rank {rank}: Trial #{trial.number}")
                lines.append(f"  Validation MAE: {trial.value:.6f}")
                lines.append(f"  Key params:")
                for key in ["ffn_hidden_dim", "ffn_num_layers", "learning_rate", "batch_size"]:
                    if key in trial.params:
                        value = trial.params[key]
                        if isinstance(value, float) and value < 0.01:
                            lines.append(f"    {key}: {value:.6f}")
                        else:
                            lines.append(f"    {key}: {value}")
                lines.append("")
        else:
            lines.append("No completed trials yet.")
            lines.append("")
        
        lines.append("=" * 80)
        
        safe_write_text(summary_file, "\n".join(lines))
        print(f"[save_all_trials_summary] Saved trials summary to {summary_file}")
    except Exception as e:
        print(f"[save_all_trials_summary] Error saving trials summary: {e}")


def build_objective(data_cfg: DataConfig,
                    base_model_cfg: ModelConfig,
                    base_training_cfg: TrainingConfig,
                    tuning_cfg: dict,
                    output_dir: Path,
                    study: optuna.Study):
    """Build the objective function for Optuna optimization."""
    
    def objective(trial: optuna.Trial):
        try:
            # ============================================================
            # MODEL ARCHITECTURE PARAMETERS
            # Only tune if specified in tuning_cfg, otherwise use base
            # ============================================================
            
            if "message_hidden_dim_choices" in tuning_cfg:
                message_hidden_dim = trial.suggest_categorical(
                    "message_hidden_dim",
                    tuning_cfg["message_hidden_dim_choices"]
                )
            else:
                message_hidden_dim = base_model_cfg.message_hidden_dim
            
            if "num_layers_min" in tuning_cfg and "num_layers_max" in tuning_cfg:
                num_layers = trial.suggest_int(
                    "num_layers",
                    tuning_cfg["num_layers_min"],
                    tuning_cfg["num_layers_max"]
                )
            else:
                num_layers = base_model_cfg.num_layers
            
            if "ffn_hidden_dim_choices" in tuning_cfg:
                ffn_hidden_dim = trial.suggest_categorical(
                    "ffn_hidden_dim",
                    tuning_cfg["ffn_hidden_dim_choices"]
                )
            else:
                ffn_hidden_dim = base_model_cfg.ffn_hidden_dim
            
            if "ffn_num_layers_min" in tuning_cfg and "ffn_num_layers_max" in tuning_cfg:
                ffn_num_layers = trial.suggest_int(
                    "ffn_num_layers",
                    tuning_cfg["ffn_num_layers_min"],
                    tuning_cfg["ffn_num_layers_max"]
                )
            else:
                ffn_num_layers = base_model_cfg.ffn_num_layers
            
            if "dropout_min" in tuning_cfg and "dropout_max" in tuning_cfg:
                dropout = trial.suggest_float(
                    "dropout",
                    tuning_cfg["dropout_min"],
                    tuning_cfg["dropout_max"]
                )
            else:
                dropout = base_model_cfg.dropout
            
            # Model-specific architecture parameters
            if base_model_cfg.model_type == "chemprop":
                if "aggregation_choices" in tuning_cfg:
                    aggregation = trial.suggest_categorical(
                        "aggregation",
                        tuning_cfg["aggregation_choices"]
                    )
                else:
                    aggregation = base_model_cfg.chemprop.aggregation
                
                activation = None  # Not used in Chemprop
            else:  # PyG model
                if "activation_choices" in tuning_cfg:
                    activation = trial.suggest_categorical(
                        "activation",
                        tuning_cfg["activation_choices"]
                    )
                else:
                    activation = base_model_cfg.pyg.activation
                
                aggregation = None  # Not used in PyG
            
            # ============================================================
            # TRAINING PARAMETERS
            # Only tune if specified in tuning_cfg, otherwise use base
            # ============================================================
            
            if "lr_min" in tuning_cfg and "lr_max" in tuning_cfg:
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    tuning_cfg["lr_min"],
                    tuning_cfg["lr_max"],
                    log=True
                )
            else:
                learning_rate = base_training_cfg.learning_rate
            
            if "batch_size_choices" in tuning_cfg:
                batch_size = trial.suggest_categorical(
                    "batch_size",
                    tuning_cfg["batch_size_choices"]
                )
            else:
                batch_size = base_training_cfg.batch_size
            
            if "optimizer_choices" in tuning_cfg:
                optimizer = trial.suggest_categorical(
                    "optimizer",
                    tuning_cfg["optimizer_choices"]
                )
            else:
                optimizer = base_training_cfg.optimizer
            
            if "weight_decay_min" in tuning_cfg and "weight_decay_max" in tuning_cfg:
                weight_decay_min = tuning_cfg["weight_decay_min"]
                weight_decay_max = tuning_cfg["weight_decay_max"]
                
                if weight_decay_min <= 0:
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
            else:
                weight_decay = base_training_cfg.weight_decay
            
            # Loss function
            if "loss_fn_choices" in tuning_cfg:
                loss_fn = trial.suggest_categorical(
                    "loss_fn",
                    tuning_cfg["loss_fn_choices"]
                )
            else:
                loss_fn = base_training_cfg.loss_fn
            
            # Huber delta (only used if loss is huber or smooth_l1)
            if loss_fn in ["huber", "smooth_l1"]:
                if "huber_delta_min" in tuning_cfg and "huber_delta_max" in tuning_cfg:
                    huber_delta = trial.suggest_float(
                        "huber_delta",
                        tuning_cfg["huber_delta_min"],
                        tuning_cfg["huber_delta_max"]
                    )
                else:
                    huber_delta = base_training_cfg.huber_delta
            else:
                huber_delta = base_training_cfg.huber_delta
            
            # ============================================================
            # SCHEDULER PARAMETERS
            # Only tune if tune_scheduler is True
            # ============================================================
            
            use_scheduler = tuning_cfg.get("tune_scheduler", False)
            
            if use_scheduler and "scheduler_type_choices" in tuning_cfg:
                scheduler_type = trial.suggest_categorical(
                    "scheduler_type",
                    tuning_cfg["scheduler_type_choices"]
                )
                
                if scheduler_type == "plateau":
                    if "scheduler_patience_min" in tuning_cfg and "scheduler_patience_max" in tuning_cfg:
                        scheduler_patience = trial.suggest_int(
                            "scheduler_patience",
                            tuning_cfg["scheduler_patience_min"],
                            tuning_cfg["scheduler_patience_max"]
                        )
                    else:
                        scheduler_patience = base_training_cfg.scheduler_patience
                    
                    if "scheduler_factor_min" in tuning_cfg and "scheduler_factor_max" in tuning_cfg:
                        scheduler_factor = trial.suggest_float(
                            "scheduler_factor",
                            tuning_cfg["scheduler_factor_min"],
                            tuning_cfg["scheduler_factor_max"]
                        )
                    else:
                        scheduler_factor = base_training_cfg.scheduler_factor
                else:
                    scheduler_patience = base_training_cfg.scheduler_patience
                    scheduler_factor = base_training_cfg.scheduler_factor
            else:
                # Use base config scheduler settings
                use_scheduler = base_training_cfg.use_scheduler
                scheduler_type = base_training_cfg.scheduler_type
                scheduler_patience = base_training_cfg.scheduler_patience
                scheduler_factor = base_training_cfg.scheduler_factor
            
            # ============================================================
            # BUILD COMPLETE CONFIG WITH SAMPLED PARAMETERS
            # This ensures Config.__post_init__ is called, which sets
            # the correct PyG dimensions automatically!
            # ============================================================
            
            if base_model_cfg.model_type == "chemprop":
                chemprop_cfg = ChemPropModelConfig(
                    aggregation=aggregation
                )
                
                model_cfg = ModelConfig(
                    model_type="chemprop",
                    message_hidden_dim=message_hidden_dim,
                    num_layers=num_layers,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_num_layers=ffn_num_layers,
                    dropout=dropout,
                    chemprop=chemprop_cfg
                )
            else:  # PyG
                # Copy base PyG config and only override tuned parameters
                pyg_cfg = PyGModelConfig(
                    node_in_dim=base_model_cfg.pyg.node_in_dim,
                    edge_in_dim=base_model_cfg.pyg.edge_in_dim,
                    edge_dim=base_model_cfg.pyg.edge_dim,
                    pool_type=base_model_cfg.pyg.pool_type,
                    pool_ratio=base_model_cfg.pyg.pool_ratio,
                    pool_num_heads=base_model_cfg.pyg.pool_num_heads,
                    pool_dim_feedforward=base_model_cfg.pyg.pool_dim_feedforward,
                    pool_num_timesteps=base_model_cfg.pyg.pool_num_timesteps,
                    deepgcn=base_model_cfg.pyg.deepgcn,
                    gnn_type=base_model_cfg.pyg.gnn_type,
                    activation=activation,  # Tuned parameter
                    num_heads=base_model_cfg.pyg.num_heads,
                    use_edge_features=base_model_cfg.pyg.use_edge_features
                )
                
                model_cfg = ModelConfig(
                    model_type="pyg",
                    message_hidden_dim=message_hidden_dim,
                    num_layers=num_layers,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_num_layers=ffn_num_layers,
                    dropout=dropout,
                    pyg=pyg_cfg
                )
            
            training_cfg = TrainingConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=tuning_cfg.get("epochs_per_trial", base_training_cfg.num_epochs),
                optimizer=optimizer,
                weight_decay=weight_decay,
                loss_fn=loss_fn,
                huber_delta=huber_delta,
                use_scheduler=use_scheduler,
                scheduler_type=scheduler_type,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
                warmup_epochs=base_training_cfg.warmup_epochs,
                early_stop_patience=tuning_cfg.get("early_stop_patience", base_training_cfg.early_stop_patience),
                monitor_metric=base_training_cfg.monitor_metric,
                monitor_mode=base_training_cfg.monitor_mode,
                checkpoint_dir=base_training_cfg.checkpoint_dir,
                save_top_k=base_training_cfg.save_top_k,
                log_dir=base_training_cfg.log_dir,
                log_every_n_steps=base_training_cfg.log_every_n_steps,
                accelerator=base_training_cfg.accelerator,
                devices=base_training_cfg.devices,
                precision=base_training_cfg.precision,
                seed=base_training_cfg.seed,
                deterministic=base_training_cfg.deterministic
            )
            
            # ============================================================
            # CREATE COMPLETE CONFIG - This triggers __post_init__!
            # ============================================================
            from ..config import Config
            
            config = Config(
                data=data_cfg,
                model=model_cfg,
                training=training_cfg,
                experiment_name=f"trial_{trial.number:04d}",
                description=f"Optuna trial {trial.number}",
                tags=["hyperparameter_tuning"]
            )
            
            print(f"[Trial {trial.number}] Config initialized with PyG dimensions: "
                  f"node_in_dim={config.model.pyg.node_in_dim if config.model.model_type == 'pyg' else 'N/A'}, "
                  f"edge_in_dim={config.model.pyg.edge_in_dim if config.model.model_type == 'pyg' else 'N/A'}")
            
            # Set seed for reproducibility (unique per trial)
            if tuning_cfg.get("seed") is not None:
                L.seed_everything(tuning_cfg["seed"] + trial.number, workers=True)
            
            # GPU allocation: Round-robin assignment across available GPUs
            n_jobs = tuning_cfg.get("n_jobs", 1)
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                trials_per_gpu = max(1, n_jobs // n_gpus)
                gpu_id = trial.number % n_gpus
                num_workers = 0 if trials_per_gpu > 1 else 2
                print(f"[Trial {trial.number}] Assigned to GPU {gpu_id} (expected {trials_per_gpu} trials/GPU)")
                
                # Override devices in config
                config.training.devices = [gpu_id]
                config.training.accelerator = "gpu"
            else:
                raise RuntimeError("No GPU available! This script requires GPU.")
            
            # ============================================================
            # USE train_from_config - reuses all the initialization logic!
            # ============================================================
            from .trainer import train_from_config
            
            # Create datamodule
            datamodule = RTDataModule(
                config=config.data,
                model_type=config.model.model_type,
                batch_size=batch_size,
                num_workers=num_workers
            )
            datamodule.prepare_data()
            datamodule.setup()
            
            # Build model using the COMPLETE config
            model = build_model(config.model)
            module = RTTrainer(
                model=model,
                model_type=config.model.model_type,
                training_config=config.training,
                rt_mean=datamodule.rt_mean,
                rt_std=datamodule.rt_std
            )
            
            # Create minimal trainer for tuning
            trainer = L.Trainer(
                max_epochs=config.training.num_epochs,
                accelerator=config.training.accelerator,
                devices=config.training.devices,
                precision=config.training.precision,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False,
                callbacks=[],
                benchmark=True,
            )
            
            # Train the model
            trainer.fit(module, datamodule=datamodule)
            
            # Get validation metrics
            try:
                val_results = trainer.validate(module, datamodule=datamodule, verbose=False)
                val_metrics = val_results[0] if isinstance(val_results, (list, tuple)) and len(val_results) > 0 else {}
            except Exception:
                val_metrics = {}
            
            # Extract MAE
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
                return float("inf")
            
            value = float(metric.item()) if isinstance(metric, torch.Tensor) else float(metric)
            
            # Save this trial's result immediately
            save_trial_result(trial, value, output_dir, study)
            
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
    
    # Load configurations
    print("[hyperparam_tune] Loading configurations...")
    data_cfg = load_yaml_to_dataclass(args.data_config, DataConfig)
    base_model_cfg = load_yaml_to_dataclass(args.model_config, ModelConfig)
    
    # Load training config
    training_config_path = args.model_config.parent / "training_config.yaml"
    if not training_config_path.exists():
        raise ValueError(f"Training config not found at {training_config_path}")
    base_training_cfg = load_yaml_to_dataclass(training_config_path, TrainingConfig)
    
    # Create a complete Config object to trigger __post_init__ and get correct dimensions
    from ..config import Config
    base_config = Config(
        data=data_cfg,
        model=base_model_cfg,
        training=base_training_cfg,
        experiment_name="base_config",
        description="Base configuration for hyperparameter tuning"
    )
    
    print(f"[hyperparam_tune] Initialized base config with correct dimensions:")
    if base_config.model.model_type == "pyg":
        print(f"  PyG node_in_dim: {base_config.model.pyg.node_in_dim}")
        print(f"  PyG edge_in_dim: {base_config.model.pyg.edge_in_dim}")
    
    # Now use the properly initialized configs from base_config
    data_cfg = base_config.data
    base_model_cfg = base_config.model
    base_training_cfg = base_config.training
    
    # Load tuning config
    tuning_cfg = yaml.safe_load(Path(args.tuning_config).read_text())
    
    print(f"[hyperparam_tune] Model type: {base_model_cfg.model_type}")
    print(f"[hyperparam_tune] Base model config: {args.model_config}")
    print(f"[hyperparam_tune] Base training config: {training_config_path}")
    
    # Determine what will be tuned
    tuned_params = []
    if "lr_min" in tuning_cfg and "lr_max" in tuning_cfg:
        tuned_params.append("learning_rate")
    if "loss_fn_choices" in tuning_cfg:
        tuned_params.append("loss_fn")
    if "message_hidden_dim_choices" in tuning_cfg:
        tuned_params.append("message_hidden_dim")
    if "num_layers_min" in tuning_cfg and "num_layers_max" in tuning_cfg:
        tuned_params.append("num_layers")
    if "ffn_hidden_dim_choices" in tuning_cfg:
        tuned_params.append("ffn_hidden_dim")
    if "ffn_num_layers_min" in tuning_cfg and "ffn_num_layers_max" in tuning_cfg:
        tuned_params.append("ffn_num_layers")
    if "dropout_min" in tuning_cfg and "dropout_max" in tuning_cfg:
        tuned_params.append("dropout")
    if "batch_size_choices" in tuning_cfg:
        tuned_params.append("batch_size")
    if "optimizer_choices" in tuning_cfg:
        tuned_params.append("optimizer")
    if "weight_decay_min" in tuning_cfg and "weight_decay_max" in tuning_cfg:
        tuned_params.append("weight_decay")
    
    print(f"[hyperparam_tune] Parameters to tune: {', '.join(tuned_params) if tuned_params else 'NONE (using base config)'}")
    
    # Create output directory
    output_dir = Path(tuning_cfg.get("output_path", "results/tuning_results")).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = tuning_cfg.get("study_name", "rt_prediction_tuning")
    run_dir = output_dir / f"{study_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[hyperparam_tune] Results will be saved to: {run_dir}")
    
    # Save configurations
    safe_write_text(run_dir / "tuning_config.yaml", 
                   yaml.dump(tuning_cfg, default_flow_style=False, sort_keys=False))
    
    model_config_dict = asdict(base_model_cfg)
    safe_write_text(run_dir / "base_model_config.yaml",
                   yaml.dump(model_config_dict, default_flow_style=False, sort_keys=False))
    
    training_config_dict = asdict(base_training_cfg)
    safe_write_text(run_dir / "base_training_config.yaml",
                   yaml.dump(training_config_dict, default_flow_style=False, sort_keys=False))
    
    # Prepare data once
    print("[hyperparam_tune] Preparing data (one-time processing)...")
    dm_temp = RTDataModule(
        config=data_cfg,
        model_type=base_model_cfg.model_type,
        batch_size=base_training_cfg.batch_size,
        num_workers=4
    )
    dm_temp.prepare_data()
    print("[hyperparam_tune] Data preparation complete. Trials will reuse processed data.")
    
    # Set global seed
    if tuning_cfg.get("seed") is not None:
        L.seed_everything(tuning_cfg["seed"], workers=True)
    
    # Create persistent storage
    storage_path = run_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"
    
    print("[hyperparam_tune] Creating Optuna study...")
    study = optuna.create_study(
        study_name=study_name,
        direction=tuning_cfg.get("direction", "minimize"),
        storage=storage,
        load_if_exists=True
    )
    
    print(f"[hyperparam_tune] Study storage: {storage_path}")
    
    # Build objective function
    obj = build_objective(data_cfg, base_model_cfg, base_training_cfg, tuning_cfg, run_dir, study)
    
    # Add callback
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"\n[Trial {trial.number}] COMPLETED - Value: {trial.value:.6f}")
            
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                print(f"[Trial {trial.number}] Best so far: {study.best_value:.6f} (Trial #{study.best_trial.number})")
                
                try:
                    save_current_best(study, run_dir)
                    save_all_trials_summary(study, run_dir)
                except Exception as e:
                    print(f"[Trial {trial.number}] Error in callback: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[Trial {trial.number}] Warning: Trial marked complete but not in completed trials list")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"\n[Trial {trial.number}] FAILED")
            if trial.values is not None:
                print(f"  Error: {trial.values}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"\n[Trial {trial.number}] PRUNED")
    
    # Run optimization
    n_jobs = tuning_cfg.get("n_jobs", 1)
    n_trials = tuning_cfg.get("trials", 100)
    
    print(f"\n[hyperparam_tune] Starting optimization:")
    print(f"  Total trials: {n_trials}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Available GPUs: {n_gpus}")
    print(f"  Trials per GPU: ~{n_jobs / n_gpus:.1f}")
    print(f"  Output directory: {run_dir}")
    print()
    
    try:
        study.optimize(
            obj,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[callback],
            catch=(Exception,)
        )
    except KeyboardInterrupt:
        print("\n[hyperparam_tune] Optimization interrupted by user")
    except Exception as e:
        print(f"\n[hyperparam_tune] Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    if completed_trials:
        print(f"Completed trials: {len(completed_trials)} / {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best val MAE: {study.best_value:.6f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        save_current_best(study, run_dir)
        save_all_trials_summary(study, run_dir)
    else:
        print("No trials completed successfully!")
    
    print(f"\nAll results saved to: {run_dir}")
    print(f"  - Individual trials: trial_XXXX.json")
    print(f"  - Current best: best_result.json and best_result.txt")
    print(f"  - Summary: trials_summary.txt")
    print(f"  - Study database: optuna_study.db (for resuming)")


if __name__ == "__main__":
    main()