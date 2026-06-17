"""
Systematic architecture comparison across splitting methods.

For each split method, a single data split is prepared once and reused for every
enabled PyG architecture. Model size can be controlled either by fixing the
message-hidden dimension (`size_mode="fixed"`) or by normalizing total parameter
counts to a GCN reference (`size_mode="normalized_params"`).
"""

import copy
import polars as pl
from pathlib import Path
from typing import Literal, Mapping

from src.config import Config, DataConfig, ModelConfig, TrainingConfig
from src.training.trainer import train_from_config, load_yaml_to_dataclass
from src.data.datamodule import RTDataModule
from src.model.model import build_model


# ----------------------------- User-facing knobs -----------------------------
split_methods = ["random", "scaffold", "butina", "mces", "mces_umap"]
architectures = {
    "gcn": True,
    "gat": True,
    "graphsage": True,
    "gin": True,
    "deepgcn": True,
    "transformer": True,
}
depth = 4
width = 256
size_mode: Literal["fixed", "normalized_params"] = "normalized_params"
num_workers = 4
batch_size = None  # None -> use base training config batch size
# -----------------------------------------------------------------------------


def count_model_params(model_cfg: ModelConfig) -> int:
    """Build a model from config and count its parameters."""
    model = build_model(model_cfg)
    return sum(p.numel() for p in model.parameters())


def choose_architecture_width(
    arch: str,
    depth: int,
    width: int,
    num_heads: int,
    base_model_cfg: ModelConfig,
    num_targets: int,
) -> tuple[int, int]:
    """
    Choose a message_hidden_dim for *arch* so its total parameter count is as
    close as possible to a GCN reference with the given *depth* and *width*.

    Returns (chosen_width, reference_param_count).
    """
    # Reference: a plain GCN with the requested depth and width.
    ref_cfg = copy.deepcopy(base_model_cfg)
    ref_cfg.model_type = "pyg"
    ref_cfg.pyg.gnn_type = "gcn"
    ref_cfg.pyg.pool_type = "mean"
    ref_cfg.num_layers = depth
    ref_cfg.message_hidden_dim = width
    ref_cfg.num_targets = num_targets
    reference_params = count_model_params(ref_cfg)

    if arch == "gcn":
        return width, reference_params

    step = num_heads if arch in ("gat", "transformer") else 1
    low = max(step, 16)
    high = 8192

    best_width = width
    best_diff = float("inf")

    def _params_for(w: int) -> int:
        cfg = copy.deepcopy(base_model_cfg)
        cfg.model_type = "pyg"
        cfg.pyg.gnn_type = arch
        cfg.pyg.pool_type = "mean"
        cfg.num_layers = depth
        cfg.message_hidden_dim = w
        cfg.num_targets = num_targets
        return count_model_params(cfg)

    # Binary search over widths to find the neighborhood with the closest count.
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = (mid // step) * step
        mid = max(mid, lo)

        params = _params_for(mid)
        diff = abs(params - reference_params)
        if diff < best_diff:
            best_diff = diff
            best_width = mid

        if params < reference_params:
            lo = mid + step
        else:
            hi = mid - step

    # Local refinement around the best width found.
    candidates = {width}
    for w in range(max(low, best_width - 4 * step), min(high, best_width + 4 * step) + 1, step):
        candidates.add(w)

    for w in candidates:
        params = _params_for(w)
        diff = abs(params - reference_params)
        if diff < best_diff:
            best_diff = diff
            best_width = w

    return best_width, reference_params


def build_result_row(
    split_method: str,
    arch: str,
    depth: int,
    requested_width: int,
    chosen_width: int,
    num_params: int,
    test_results: Mapping[str, float | int] | None,
) -> dict:
    """Assemble a single CSV result row from a finished run."""
    row: dict = {
        "split_method": split_method,
        "architecture": arch,
        "depth": depth,
        "requested_width": requested_width,
        "chosen_width": chosen_width,
        "num_params": num_params,
    }

    if test_results:
        row["test/mae"] = test_results.get("test/mae_mean", test_results.get("test/mae", float("nan")))
        row["test/rmse"] = test_results.get("test/rmse_mean", test_results.get("test/rmse", float("nan")))
        row["test/r2"] = test_results.get("test/r2_mean", test_results.get("test/r2", float("nan")))
        # Copy per-target metrics when present (e.g., test/mae_rt, test/mae_ccs).
        for key, value in test_results.items():
            if key.startswith("test/") and key not in row:
                row[key] = value
    else:
        row["test/mae"] = float("nan")
        row["test/rmse"] = float("nan")
        row["test/r2"] = float("nan")

    return row


def print_pivot_tables(df: pl.DataFrame) -> None:
    """Print architecture-vs-split-method pivot tables for each metric."""
    for metric in ["test/mae", "test/rmse", "test/r2", "num_params"]:
        if metric not in df.columns:
            continue

        pivot = df.pivot(index="architecture", on="split_method", values=metric)
        # Ensure every split method is present as a column.
        for method in split_methods:
            if method not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(None).alias(method))

        pivot = pivot.select(["architecture"] + split_methods)

        print(f"\n--- {metric} ---")
        header = "| " + " | ".join(pivot.columns) + " |"
        separator = "|---" * len(pivot.columns) + "|"
        print(header)
        print(separator)
        for row_data in pivot.iter_rows():
            cells = []
            for val in row_data:
                if val is None:
                    cells.append("NaN")
                elif isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            print("| " + " | ".join(cells) + " |")


def main() -> None:
    """Run the systematic architecture comparison."""
    base_data_config = load_yaml_to_dataclass(Path("configs/data_config.yaml"), DataConfig)
    base_model_config = load_yaml_to_dataclass(Path("configs/model_config.yaml"), ModelConfig)
    base_training_config = load_yaml_to_dataclass(Path("configs/training_config.yaml"), TrainingConfig)

    num_targets = len(base_data_config.target_columns)
    num_heads = base_model_config.pyg.num_heads

    results_file = Path("systematic_architecture_comparison_results.csv")
    if results_file.exists():
        print(f"Loading existing results from {results_file}")
        existing_df = pl.read_csv(results_file)
        results = existing_df.to_dicts()
        completed = {
            (row["split_method"], row["architecture"], row["depth"], row["requested_width"])
            for row in results
        }
    else:
        results = []
        completed = set()

    effective_batch_size = batch_size or base_training_config.batch_size

    for split_method in split_methods:
        print(f"\n===== Split method: {split_method} =====")

        data_cfg = copy.deepcopy(base_data_config)
        data_cfg.split_method = split_method  # type: ignore[assignment]

        # Prepare the split once and reuse it for every architecture.
        dm = RTDataModule(
            config=data_cfg,
            model_type="pyg",
            batch_size=effective_batch_size,
            num_workers=num_workers,
        )
        print(f"[ArchitectureComparison] Preparing split for {split_method}...")
        dm.prepare_data()

        for arch, enabled in architectures.items():
            if not enabled:
                continue

            key = (split_method, arch, depth, width)
            if key in completed:
                print(f"Skipping completed experiment: {key}")
                continue

            print(f"\n[ArchitectureComparison] Running architecture: {arch}")

            model_cfg = copy.deepcopy(base_model_config)
            model_cfg.model_type = "pyg"
            model_cfg.pyg.gnn_type = arch  # type: ignore[assignment]
            model_cfg.num_layers = depth
            model_cfg.num_targets = num_targets

            # AttentiveFP readout is only valid with deepgcn; fall back to mean for
            # the comparison to keep all architectures on equal footing.
            if arch != "deepgcn":
                model_cfg.pyg.pool_type = "mean"  # type: ignore[assignment]

            if size_mode == "fixed":
                if arch in ("gat", "transformer") and width % num_heads != 0:
                    raise ValueError(
                        f"For architecture '{arch}', width ({width}) must be divisible "
                        f"by num_heads ({num_heads}) in fixed mode."
                    )
                chosen_width = width
                model_cfg.message_hidden_dim = chosen_width
                num_params = count_model_params(model_cfg)
                print(f"[ArchitectureComparison] Fixed width: {chosen_width}, params: {num_params}")
            else:  # normalized_params
                chosen_width, reference_params = choose_architecture_width(
                    arch, depth, width, num_heads, base_model_config, num_targets
                )
                model_cfg.message_hidden_dim = chosen_width
                num_params = count_model_params(model_cfg)
                print(
                    f"[ArchitectureComparison] Normalized width: {chosen_width} "
                    f"(reference GCN params: {reference_params}, this arch: {num_params})"
                )

            config = Config(
                data=data_cfg,
                model=model_cfg,
                training=copy.deepcopy(base_training_config),
                experiment_name=f"arch_comp_{split_method}_{arch}_d{depth}_w{width}",
                description=f"Architecture comparison: {arch} on {split_method} split",
                tags=["architecture_comparison"],
            )

            try:
                _, _, _, test_results_list = train_from_config(config)
                test_results = test_results_list[0] if test_results_list else None
            except Exception as e:
                print(f"[ArchitectureComparison] Experiment failed for {arch} on {split_method}: {e}")
                test_results = None

            row = build_result_row(
                split_method=split_method,
                arch=arch,
                depth=depth,
                requested_width=width,
                chosen_width=chosen_width,
                num_params=num_params,
                test_results=test_results,
            )
            results.append(row)
            completed.add(key)

            # Incremental save after every architecture.
            df = pl.DataFrame(results)
            df.write_csv(results_file)
            print(f"[ArchitectureComparison] Saved results to {results_file}")

    if not results:
        print("No results were generated.")
        return

    final_df = pl.DataFrame(results)
    print_pivot_tables(final_df)


if __name__ == "__main__":
    main()
