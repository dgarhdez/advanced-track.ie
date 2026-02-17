from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - exercised when polars isn't installed
    pl = None  # type: ignore[assignment]

from dashboard.financial_pipeline import run_duckdb_query, sanitize_schema


def _to_pandas_frame(df: Any) -> pd.DataFrame:
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame(df)


def _to_polars_frame(df: Any) -> "pl.DataFrame":
    if pl is None:
        raise RuntimeError("polars is not installed.")
    if isinstance(df, pl.DataFrame):
        return df.clone()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return pl.DataFrame(df)


def build_hr_filters(
    alias: str = "e",
    start_date: date | None = None,
    end_date: date | None = None,
    departments: Iterable[str] | None = None,
    jobs: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if start_date is not None:
        params["start_hiredate"] = datetime.combine(start_date, time.min)
        clauses.append(f"{alias}.hiredate >= :start_hiredate")

    if end_date is not None:
        params["end_hiredate"] = datetime.combine(end_date + timedelta(days=1), time.min)
        clauses.append(f"{alias}.hiredate < :end_hiredate")

    department_values = [item for item in (departments or []) if item]
    if department_values:
        placeholders: list[str] = []
        for index, value in enumerate(department_values):
            key = f"dept_{index}"
            params[key] = value
            placeholders.append(f":{key}")
        clauses.append(f"{alias}.workdept IN ({', '.join(placeholders)})")

    job_values = [item for item in (jobs or []) if item]
    if job_values:
        placeholders = []
        for index, value in enumerate(job_values):
            key = f"job_{index}"
            params[key] = value
            placeholders.append(f":{key}")
        clauses.append(f"{alias}.job IN ({', '.join(placeholders)})")

    gender_values = [item for item in (genders or []) if item]
    if gender_values:
        placeholders = []
        for index, value in enumerate(gender_values):
            key = f"gender_{index}"
            params[key] = value
            placeholders.append(f":{key}")
        clauses.append(f"{alias}.gender IN ({', '.join(placeholders)})")

    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def get_hr_filter_options(
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    sanitize_schema(schema)
    bounds_query = """
        SELECT
            MIN(CAST(hiredate AS DATE)) AS min_date,
            MAX(CAST(hiredate AS DATE)) AS max_date
        FROM EMPLOYEE
    """
    dept_query = """
        SELECT deptno, deptname
        FROM DEPARTMENT
        ORDER BY deptno
    """
    job_query = """
        SELECT DISTINCT job
        FROM EMPLOYEE
        WHERE job IS NOT NULL
        ORDER BY job
    """
    gender_query = """
        SELECT DISTINCT gender
        FROM EMPLOYEE
        WHERE gender IS NOT NULL
        ORDER BY gender
    """

    bounds_df = _to_pandas_frame(
        run_duckdb_query(
            bounds_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    dept_df = _to_pandas_frame(
        run_duckdb_query(
            dept_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    job_df = _to_pandas_frame(
        run_duckdb_query(
            job_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )
    gender_df = _to_pandas_frame(
        run_duckdb_query(
            gender_query,
            env_path=env_path,
            database_path=database_path,
            as_polars=pl is not None,
        )
    )

    bounds_df.columns = [column.lower() for column in bounds_df.columns]
    dept_df.columns = [column.lower() for column in dept_df.columns]
    job_df.columns = [column.lower() for column in job_df.columns]
    gender_df.columns = [column.lower() for column in gender_df.columns]

    return {
        "min_date": pd.to_datetime(bounds_df.loc[0, "min_date"]).date(),
        "max_date": pd.to_datetime(bounds_df.loc[0, "max_date"]).date(),
        "departments": [str(value) for value in dept_df["deptno"].dropna().tolist()],
        "jobs": [str(value) for value in job_df["job"].dropna().tolist()],
        "genders": [str(value) for value in gender_df["gender"].dropna().tolist()],
    }


def extract_hr_base_data(
    start_date: date,
    end_date: date,
    departments: Iterable[str] | None = None,
    jobs: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    env_path: str = ".env",
    schema: str | None = None,
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    sanitize_schema(schema)
    where_clause, params = build_hr_filters(
        alias="e",
        start_date=start_date,
        end_date=end_date,
        departments=departments,
        jobs=jobs,
        genders=genders,
    )
    query = f"""
        SELECT
            e.empno,
            e.firstnme,
            e.lastname,
            e.workdept AS deptno,
            d.deptname,
            d.location,
            COALESCE(d.budget, 0) AS dept_budget,
            e.job,
            e.gender,
            e.hiredate,
            e.birthdate,
            COALESCE(e.salary, 0) AS salary,
            COALESCE(e.bonus, 0) AS bonus,
            COALESCE(e.comm, 0) AS comm,
            e.is_external,
            e.is_parttime,
            fc.fleet_crew_capacity
        FROM EMPLOYEE e
        LEFT JOIN DEPARTMENT d
            ON e.workdept = d.deptno
        CROSS JOIN (
            SELECT SUM(COALESCE(crew_members, 0)) AS fleet_crew_capacity
            FROM AIRPLANES
        ) fc
        {where_clause}
    """
    df = run_duckdb_query(
        query,
        params=params,
        env_path=env_path,
        database_path=database_path,
        as_polars=pl is not None,
    )
    return _to_pandas_frame(normalize_hr_base_data(df))


def normalize_hr_base_data(
    df: pd.DataFrame | "pl.DataFrame",
) -> pd.DataFrame | "pl.DataFrame":
    if pl is not None and isinstance(df, pl.DataFrame):
        normalized = df.rename({column: column.lower() for column in df.columns})
        for column in ("salary", "bonus", "comm", "dept_budget", "fleet_crew_capacity"):
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias(column)
                )
        for column in ("hiredate", "birthdate"):
            if column in normalized.columns:
                normalized = normalized.with_columns(
                    pl.col(column).cast(pl.Datetime, strict=False).alias(column)
                )
        return normalized

    normalized_pd = df.copy()
    normalized_pd.columns = [column.lower() for column in normalized_pd.columns]
    for column in ("salary", "bonus", "comm", "dept_budget", "fleet_crew_capacity"):
        if column in normalized_pd.columns:
            normalized_pd[column] = pd.to_numeric(normalized_pd[column], errors="coerce").fillna(0.0)
    for column in ("hiredate", "birthdate"):
        if column in normalized_pd.columns:
            normalized_pd[column] = pd.to_datetime(normalized_pd[column], errors="coerce")
    return normalized_pd


def _empty_hr_views() -> dict[str, Any]:
    return {
        "kpis": {
            "total_headcount": 0,
            "total_salary": 0.0,
            "total_compensation": 0.0,
            "total_budget": 0.0,
            "budget_utilization": 0.0,
            "operational_headcount": 0,
            "fleet_crew_capacity": 0.0,
            "crew_to_operational_ratio": 0.0,
        },
        "department_summary": pd.DataFrame(),
        "job_headcount": pd.DataFrame(),
        "gender_distribution": pd.DataFrame(),
        "hire_trend": pd.DataFrame(),
    }


def _compute_hr_views_pandas(base_df: pd.DataFrame) -> dict[str, Any]:
    working = _to_pandas_frame(normalize_hr_base_data(base_df))
    if working.empty:
        return _empty_hr_views()

    working["total_compensation"] = (
        pd.to_numeric(working["salary"], errors="coerce").fillna(0.0)
        + pd.to_numeric(working["bonus"], errors="coerce").fillna(0.0)
        + pd.to_numeric(working["comm"], errors="coerce").fillna(0.0)
    )

    department_summary = (
        working.groupby(["deptno", "deptname", "location"], as_index=False)
        .agg(
            headcount=("empno", "nunique"),
            salary_total=("salary", "sum"),
            compensation_total=("total_compensation", "sum"),
            avg_salary=("salary", "mean"),
            dept_budget=("dept_budget", "max"),
        )
        .sort_values("compensation_total", ascending=False)
    )
    department_summary["budget_utilization"] = department_summary.apply(
        lambda row: float(row["compensation_total"]) / float(row["dept_budget"])
        if float(row["dept_budget"]) > 0
        else 0.0,
        axis=1,
    )

    job_headcount = (
        working.groupby(["job"], as_index=False)
        .agg(headcount=("empno", "nunique"), avg_salary=("salary", "mean"))
        .sort_values("headcount", ascending=False)
    )
    gender_distribution = (
        working.groupby(["gender"], as_index=False)
        .agg(headcount=("empno", "nunique"))
        .sort_values("headcount", ascending=False)
    )

    hires = working.copy()
    hires["hire_month"] = pd.to_datetime(hires["hiredate"], errors="coerce").dt.to_period("M").astype(str)
    hire_trend = (
        hires.groupby(["hire_month"], as_index=False)
        .agg(hires=("empno", "nunique"))
        .sort_values("hire_month")
    )

    job_series = working["job"].astype(str).str.upper()
    dept_series = working["deptname"].astype(str).str.upper()
    operational_mask = (
        job_series.str.contains(r"PILOT|ATTENDANT|CREW|FLIGHT", na=False)
        | dept_series.str.contains(r"OPER|FLIGHT|CREW|OPS", na=False)
    )
    operational_headcount = int(working.loc[operational_mask, "empno"].nunique())
    fleet_crew_capacity = float(pd.to_numeric(working["fleet_crew_capacity"], errors="coerce").max())

    total_headcount = int(working["empno"].nunique())
    total_salary = float(pd.to_numeric(working["salary"], errors="coerce").fillna(0.0).sum())
    total_comp = float(pd.to_numeric(working["total_compensation"], errors="coerce").fillna(0.0).sum())
    total_budget = float(pd.to_numeric(department_summary["dept_budget"], errors="coerce").fillna(0.0).sum())

    return {
        "kpis": {
            "total_headcount": total_headcount,
            "total_salary": total_salary,
            "total_compensation": total_comp,
            "total_budget": total_budget,
            "budget_utilization": total_comp / total_budget if total_budget else 0.0,
            "operational_headcount": operational_headcount,
            "fleet_crew_capacity": fleet_crew_capacity,
            "crew_to_operational_ratio": (
                fleet_crew_capacity / operational_headcount if operational_headcount else 0.0
            ),
        },
        "department_summary": department_summary,
        "job_headcount": job_headcount,
        "gender_distribution": gender_distribution,
        "hire_trend": hire_trend,
    }


def _compute_hr_views_polars(base_df: pd.DataFrame | "pl.DataFrame") -> dict[str, Any]:
    if pl is None:
        return _compute_hr_views_pandas(_to_pandas_frame(base_df))

    working = normalize_hr_base_data(_to_polars_frame(base_df))
    if working.is_empty():
        return _empty_hr_views()

    working = working.with_columns(
        (pl.col("salary") + pl.col("bonus") + pl.col("comm")).alias("total_compensation")
    )
    department_summary = (
        working.group_by(["deptno", "deptname", "location"])
        .agg(
            [
                pl.col("empno").n_unique().alias("headcount"),
                pl.col("salary").sum().alias("salary_total"),
                pl.col("total_compensation").sum().alias("compensation_total"),
                pl.col("salary").mean().alias("avg_salary"),
                pl.col("dept_budget").max().alias("dept_budget"),
            ]
        )
        .with_columns(
            pl.when(pl.col("dept_budget") == 0)
            .then(0.0)
            .otherwise(pl.col("compensation_total") / pl.col("dept_budget"))
            .alias("budget_utilization")
        )
        .sort("compensation_total", descending=True)
    )

    job_headcount = (
        working.group_by("job")
        .agg([pl.col("empno").n_unique().alias("headcount"), pl.col("salary").mean().alias("avg_salary")])
        .sort("headcount", descending=True)
    )
    gender_distribution = (
        working.group_by("gender").agg(pl.col("empno").n_unique().alias("headcount")).sort("headcount", descending=True)
    )
    hire_trend = (
        working.with_columns(pl.col("hiredate").dt.strftime("%Y-%m").alias("hire_month"))
        .group_by("hire_month")
        .agg(pl.col("empno").n_unique().alias("hires"))
        .sort("hire_month")
    )

    operational_mask = (
        pl.col("job")
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.contains("PILOT|ATTENDANT|CREW|FLIGHT")
        | pl.col("deptname").cast(pl.Utf8).str.to_uppercase().str.contains("OPER|FLIGHT|CREW|OPS")
    )
    operational_headcount = int(working.filter(operational_mask)["empno"].n_unique())
    fleet_crew_capacity = float(working["fleet_crew_capacity"].max())

    total_headcount = int(working["empno"].n_unique())
    total_salary = float(working["salary"].sum())
    total_comp = float(working["total_compensation"].sum())
    total_budget = float(department_summary["dept_budget"].sum())

    return {
        "kpis": {
            "total_headcount": total_headcount,
            "total_salary": total_salary,
            "total_compensation": total_comp,
            "total_budget": total_budget,
            "budget_utilization": total_comp / total_budget if total_budget else 0.0,
            "operational_headcount": operational_headcount,
            "fleet_crew_capacity": fleet_crew_capacity,
            "crew_to_operational_ratio": (
                fleet_crew_capacity / operational_headcount if operational_headcount else 0.0
            ),
        },
        "department_summary": department_summary.to_pandas(),
        "job_headcount": job_headcount.to_pandas(),
        "gender_distribution": gender_distribution.to_pandas(),
        "hire_trend": hire_trend.to_pandas(),
    }


def compute_hr_views(
    base_df: pd.DataFrame | "pl.DataFrame",
) -> dict[str, Any]:
    if pl is not None:
        return _compute_hr_views_polars(base_df=base_df)
    return _compute_hr_views_pandas(_to_pandas_frame(base_df))
