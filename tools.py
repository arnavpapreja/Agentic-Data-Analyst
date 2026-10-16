"""Safe, reusable tabular-analysis tools for the agent.

These functions operate on the active pandas DataFrame. They never execute
model-generated Python or SQL; the model can only call the allow-listed tools.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _clean(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-friendly Python values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _check_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Unknown column '{column}'. Available columns: {available}")


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return dimensions, column types, memory use, and a small data preview."""
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": [
            {"name": str(col), "dtype": str(df[col].dtype), "non_null": int(df[col].notna().sum()),
             "missing": int(df[col].isna().sum()), "unique_values": int(df[col].nunique(dropna=True))}
            for col in df.columns
        ],
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "preview": _clean(df.head(5).to_dict(orient="records")),
    }


def column_profile(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Profile one column with numeric distribution or categorical frequencies."""
    _check_column(df, column)
    series = df[column]
    out: dict[str, Any] = {
        "column": column,
        "dtype": str(series.dtype),
        "row_count": int(len(series)),
        "missing_count": int(series.isna().sum()),
        "missing_percent": round(float(series.isna().mean() * 100), 2) if len(series) else 0.0,
        "unique_non_null": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe(percentiles=[.25, .5, .75]).to_dict()
        out["kind"] = "numeric"
        out["statistics"] = _clean(desc)
        out["zero_count"] = int((series == 0).sum())
        out["negative_count"] = int((series < 0).sum())
    else:
        out["kind"] = "categorical_or_text"
        out["top_values"] = [
            {"value": _clean(value), "count": int(count)}
            for value, count in series.value_counts(dropna=False).head(10).items()
        ]
        out["examples"] = _clean(series.dropna().astype(str).drop_duplicates().head(5).tolist())
    return out


def missing_data_report(df: pd.DataFrame) -> dict[str, Any]:
    """Rank columns by missing data and report complete/duplicate rows."""
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append({"column": str(col), "missing_count": missing,
                     "missing_percent": round(float(missing / len(df) * 100), 2) if len(df) else 0.0})
    rows.sort(key=lambda x: x["missing_count"], reverse=True)
    return {
        "rows": int(len(df)),
        "fully_complete_rows": int(df.notna().all(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_ranked_by_missing": rows,
    }


def value_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> dict[str, Any]:
    """Return frequency distribution for a categorical or numeric column."""
    _check_column(df, column)
    top_n = max(1, min(int(top_n), 50))
    series = df[column]
    counts = series.value_counts(dropna=False).head(top_n)
    return {"column": column, "unique_non_null": int(series.nunique(dropna=True)),
            "frequencies": [{"value": _clean(k), "count": int(v),
                             "percent_of_rows": round(float(v / len(df) * 100), 2) if len(df) else 0.0}
                            for k, v in counts.items()]}


def grouped_aggregation(df: pd.DataFrame, group_by: str, metric: str, aggregation: str = "sum") -> dict[str, Any]:
    """Aggregate a metric by groups; supports sum, mean, median, min, max, count, nunique."""
    _check_column(df, group_by)
    _check_column(df, metric)
    allowed = {"sum", "mean", "median", "min", "max", "count", "nunique"}
    aggregation = aggregation.lower().strip()
    if aggregation not in allowed:
        raise ValueError(f"Unsupported aggregation '{aggregation}'. Choose from: {', '.join(sorted(allowed))}")
    if aggregation in {"sum", "mean", "median", "min", "max"} and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"'{metric}' is not numeric. Choose a numeric metric or use count/nunique.")
    grouped = df.groupby(group_by, dropna=False)[metric]
    result = getattr(grouped, aggregation)().sort_values(ascending=False, na_position="last")
    return {"group_by": group_by, "metric": metric, "aggregation": aggregation,
            "group_count": int(len(result)),
            "results": [{"group": _clean(group), "value": _clean(value)} for group, value in result.head(50).items()]}


def filter_rows(df: pd.DataFrame, column: str, operator: str, value: str) -> dict[str, Any]:
    """Filter rows using a safe allow-list of operators; return count and sample only."""
    _check_column(df, column)
    ops = {"==", "!=", ">", ">=", "<", "<=", "contains", "in"}
    if operator not in ops:
        raise ValueError(f"Unsupported operator '{operator}'. Choose from: {', '.join(sorted(ops))}")
    series = df[column]
    try:
        if operator == "contains":
            mask = series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        elif operator == "in":
            values = [v.strip() for v in str(value).split(",")]
            if pd.api.types.is_numeric_dtype(series):
                converted = [float(v) for v in values]
                mask = series.isin(converted)
            else:
                mask = series.astype(str).isin(values)
        elif pd.api.types.is_numeric_dtype(series):
            compare_value: Any = float(value)
            if operator == "==": mask = series == compare_value
            elif operator == "!=": mask = series != compare_value
            elif operator == ">": mask = series > compare_value
            elif operator == ">=": mask = series >= compare_value
            elif operator == "<": mask = series < compare_value
            else: mask = series <= compare_value
        elif pd.api.types.is_bool_dtype(series):
            compare_value = str(value).lower() in {"true", "1", "yes"}
            if operator == "==": mask = series == compare_value
            elif operator == "!=": mask = series != compare_value
            else: raise ValueError("Ordered comparisons are not supported for boolean columns.")
        else:
            if operator == "==": mask = series.astype(str) == str(value)
            elif operator == "!=": mask = series.astype(str) != str(value)
            elif operator == ">": mask = series.astype(str) > str(value)
            elif operator == ">=": mask = series.astype(str) >= str(value)
            elif operator == "<": mask = series.astype(str) < str(value)
            else: mask = series.astype(str) <= str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not compare '{value}' with column '{column}': {exc}") from exc
    filtered = df.loc[mask.fillna(False)]
    return {"condition": {"column": column, "operator": operator, "value": value},
            "matching_rows": int(len(filtered)), "total_rows": int(len(df)),
            "percent_matching": round(float(len(filtered) / len(df) * 100), 2) if len(df) else 0.0,
            "sample_rows": _clean(filtered.head(10).to_dict(orient="records")),
            "sample_limit": 10}


def correlation_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate Pearson correlations for numeric columns and rank strongest pairs."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"numeric_columns": list(map(str, numeric.columns)), "pairs": [],
                "message": "At least two numeric columns are required for correlation analysis."}
    corr = numeric.corr(method="pearson")
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"column_1": str(left), "column_2": str(right), "pearson_r": round(float(value), 4)})
    pairs.sort(key=lambda row: abs(row["pearson_r"]), reverse=True)
    return {"numeric_columns": list(map(str, numeric.columns)), "method": "Pearson correlation",
            "pairs": pairs[:30], "caution": "Correlation indicates association, not causation."}


def build_tool_specs() -> list[dict[str, Any]]:
    """JSON schemas converted into Gemini function declarations for model-guided function calling."""
    return [
        {"type": "function", "function": {"name": "dataset_overview", "description": "Inspect dataset dimensions, column names/types, missing values, duplicates, and a small preview. Use when starting an analysis or asking general questions about the dataset.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "column_profile", "description": "Analyze one column. Numeric columns return descriptive statistics; text/categorical columns return common values and examples.", "parameters": {"type": "object", "properties": {"column": {"type": "string", "description": "Exact column name from the dataset"}}, "required": ["column"]}}},
        {"type": "function", "function": {"name": "missing_data_report", "description": "Find missing values, completeness, and duplicate rows across the whole dataset.", "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "value_counts", "description": "Count the frequency of each value in a column; useful for category distribution and most common labels.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "top_n": {"type": "integer", "description": "Maximum number of values to return (1-50)"}}, "required": ["column"]}}},
        {"type": "function", "function": {"name": "grouped_aggregation", "description": "Aggregate a metric by groups, such as total revenue by region or average profit by product. Supports sum, mean, median, min, max, count, nunique.", "parameters": {"type": "object", "properties": {"group_by": {"type": "string", "description": "Column defining groups"}, "metric": {"type": "string", "description": "Column to aggregate"}, "aggregation": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"]}}, "required": ["group_by", "metric", "aggregation"]}}},
        {"type": "function", "function": {"name": "filter_rows", "description": "Filter records with a safe comparison and return the number matching plus up to ten example rows.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "in"]}, "value": {"type": "string", "description": "Comparison value; for 'in', comma-separated values"}}, "required": ["column", "operator", "value"]}}},
        {"type": "function", "function": {"name": "correlation_analysis", "description": "Find strong Pearson correlations between numeric columns. This does not prove causation.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]
