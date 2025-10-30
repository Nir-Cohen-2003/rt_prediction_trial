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
    split_methods = ["random", "mces"] #"random", "scaffold", 
    depths = [8] # 2, , 
    widths = [ 512, 1024, 2048] #32, 64, 128,
    
    base_data_config = load_yaml_to_dataclass(Path("configs/data_config.yaml"), DataConfig)
    base_model_config = load_yaml_to_dataclass(Path("configs/model_config.yaml"), ModelConfig)
    base_training_config = load_yaml_to_dataclass(Path("configs/training_config.yaml"), TrainingConfig)

    # Load existing results if available
    results_file = Path("systematic_comparison_results.csv")
    if results_file.exists():
        print(f"Loading existing results from {results_file}")
        existing_df = pl.read_csv(results_file)
        results = existing_df.to_dicts()
        # Create set of already-run experiments to skip duplicates
        completed_experiments = {
            (row["split_method"], row["depth"], row["width"]) 
            for row in results
        }
    else:
        results = []
        completed_experiments = set()

    for split_method in split_methods:
        for depth, width in product(depths, widths):
            # Skip if already completed
            if (split_method, depth, width) in completed_experiments:
                print(f"Skipping already completed experiment: split={split_method}, depth={depth}, width={width}")
                continue
                
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
                    completed_experiments.add((split_method, depth, width))
                    
                    # Save intermediate results (overwrites with full dataset)
                    df = pl.DataFrame(results)
                    df.write_csv(results_file)
                    print(f"Saved results to {results_file}")

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
        pl.concat_str(
            pl.lit("d="),
            pl.col("depth").cast(pl.String),
            pl.lit(", w="),
            pl.col("width").cast(pl.String),
            pl.lit(" ("),
            pl.col("num_params").cast(pl.String),
            pl.lit(")")
        ).alias("size_label")
    )

    for metric in ["test/mae", "test/rmse", "test/r2"]:
        pivot_table = df.pivot(
            index="size_label",
            on="split_method",
            values=metric
        )
        # Ensure all split methods are present as columns
        existing_cols = set(pivot_table.columns) - {"size_label"}
        for method in split_methods:
            if method not in existing_cols:
                pivot_table = pivot_table.with_columns(pl.lit(None).alias(method))
        
        # Reorder columns
        pivot_table = pivot_table.select(["size_label"] + split_methods)
        
        print(f"\n--- Results for {metric} ---")
        
        # Format float columns for markdown output
        formatted_df = pivot_table.with_columns(
            [pl.col(col).cast(pl.String).fill_null("NaN") for col in split_methods]
        )
        
        # Convert to markdown table
        header = "| " + " | ".join(formatted_df.columns) + " |"
        separator = "|---" * len(formatted_df.columns) + "|"
        
        rows = []
        for row_data in formatted_df.iter_rows():
            rows.append("| " + " | ".join(str(val) for val in row_data) + " |")
        
        print(header)
        print(separator)
        print("\n".join(rows))
        print("\n")


if __name__ == "__main__":
    main()
