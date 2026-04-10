"""Export benchmark results to LaTeX, Markdown, and CSV."""

from __future__ import annotations

import pandas as pd


def to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
) -> str:
    """Convert results DataFrame to a LaTeX booktabs table string.

    Parameters
    ----------
    df : pd.DataFrame
        Results from BenchmarkRunner.run() with columns:
        policy, forecaster, echelon, metric, value.
    caption : str
        Table caption.
    label : str
        LaTeX label for referencing.

    Returns
    -------
    str
        LaTeX table source.
    """
    pivot = df.pivot_table(
        index=["policy", "echelon"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    latex = pivot.to_latex(float_format="%.3f")

    if caption or label:
        header = "\\begin{table}[htbp]\n\\centering\n"
        if caption:
            header += f"\\caption{{{caption}}}\n"
        if label:
            header += f"\\label{{{label}}}\n"
        latex = header + latex + "\n\\end{table}"

    return latex


def to_markdown(df: pd.DataFrame) -> str:
    """Convert results DataFrame to GitHub-flavored markdown table.

    Parameters
    ----------
    df : pd.DataFrame
        Results from BenchmarkRunner.run().

    Returns
    -------
    str
        Markdown table string.
    """
    pivot = df.pivot_table(
        index=["policy", "echelon"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    try:
        return pivot.to_markdown(floatfmt=".3f")
    except ImportError:
        # Fallback if tabulate is not installed
        return pivot.to_string(float_format="%.3f")
