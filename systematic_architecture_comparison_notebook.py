import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def imports():
    import importlib
    from pathlib import Path
    import polars as pl
    import systematic_architecture_comparison as sac
    return importlib, Path, pl, sac


@app.cell
def run_comparison(sac):
    sac.main()
    return


@app.cell
def load_results(Path, pl):
    results_file = Path("systematic_architecture_comparison_results.csv")
    df = pl.read_csv(results_file) if results_file.exists() else pl.DataFrame()
    return df, results_file


@app.cell
def display_table(df):
    df
    return


@app.cell
def display_pivots(df, sac):
    if not df.is_empty():
        sac.print_pivot_tables(df)
    return


if __name__ == "__main__":
    app.run()
