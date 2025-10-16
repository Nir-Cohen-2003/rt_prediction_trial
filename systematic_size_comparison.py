
import yaml
import copy
import polars as pl
from pathlib import Path
from itertools import product

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.training.trainer import train_from_config, load_yaml_to_dataclass

def main():
    """
    Systematically compares different data splitting methods and model sizes
    for the deepgcn model.
    """
    split_methods = ["random", "scaffold", "butina", "mces"]
    depths = [2, 4, 8, 16]
    widths = [32, 64, 128, 256]
    
    base_data_config = load_yaml_to_dataclass(Path("configs/data_config.yaml"), DataConfig)
    base_model_config = load_yaml_to_dataclass(Path("configs/model_config.yaml"), ModelConfig)
    base_training_config = load_yaml_to_dataclass(Path("configs/training_config.yaml"), TrainingConfig)

    results = []

    for split_method in split_methods:
        for depth, width in product(depths, widths):
            print(f"Running experiment with split: {split_method}, depth: {depth}, width: {width}")

            config = Config(
                data=copy.deepcopy(base_data_config),
                model=copy.deepcopy(base_model_config),
                training=copy.deepcopy(base_training_config),
                experiment_name=f"sys_comp_{split_method}_d{depth}_w{width}",
                description=f"Systematic comparison for {split_method} split, deepgcn model with depth {depth} and width {width}",
                tags=["systematic_comparison", "deepgcn"]
            )

            # --- Modify config for the current run ---
            # Data
            config.data.split_method = split_method
            
            # Model
            config.model.model_type = "pyg"
            config.model.pyg.gnn_type = "deepgcn"
            config.model.num_layers = depth
            config.model.message_hidden_dim = width
            config.model.ffn_hidden_dim = width

            try:
                trainer, module, datamodule, test_results = train_from_config(config)
                
                if test_results:
                    num_params = sum(p.numel() for p in module.model.parameters())
                    
                    result_row = {
                        "split_method": split_method,
                        "depth": depth,
                        "width": width,
                        "num_params": num_params,
                        "test/mae": test_results[0].get("test/mae", float("nan")),
                        "test/rmse": test_results[0].get("test/rmse", float("nan")),
                        "test/r2": test_results[0].get("test/r2", float("nan")),
                    }
                    results.append(result_row)
                    
                    # Save intermediate results
                    df = pl.DataFrame(results)
                    df.write_csv("systematic_comparison_results.csv")

            except Exception as e:
                print(f"--- Experiment failed for split: {split_method}, depth: {depth}, width: {width} ---")
                print(e)
                print("--------------------------------------------------------------------")


    # --- Process and print final results ---
    if not results:
        print("No results were generated.")
        return

    df = pl.DataFrame(results)
    df = df.sort("num_params")
    df = df.with_columns(
        pl.format("d={}, w={} ({:,})", pl.col("depth"), pl.col("width"), pl.col("num_params")).alias("size_label")
    )

    for metric in ["test/mae", "test/rmse", "test/r2"]:
        pivot_table = df.pivot(
            index="size_label",
            columns="split_method",
            values=metric
        )
        # Ensure all split methods are present as columns, filling with nulls if not
        # And reorder them as per split_methods list
        pivot_table = pivot_table.select([pl.col("size_label")] + split_methods)
        
        print(f"\n--- Results for {metric} ---")
        
        # Format float columns for markdown output
        formatted_df = pivot_table.with_columns(
            pl.col(pl.Float64).map_elements(lambda x: f"{x:.4f}" if not pl.is_null(x) else "NaN", return_dtype=pl.String)
        )
        
        # Convert to string and print as markdown
        # Polars to_string() doesn't directly support markdown, so we'll format manually
        header = "| " + " | ".join(formatted_df.columns) + " |"
        separator = "|---" * len(formatted_df.columns) + "|"
        
        rows = []
        for row_data in formatted_df.iter_rows():
            rows.append("| " + " | ".join(row_data) + " |")
        
        print(header)
        print(separator)
        print("\n".join(rows))
        print("\n")


if __name__ == "__main__":
    main()
