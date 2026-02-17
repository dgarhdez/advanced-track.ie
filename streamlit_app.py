from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.commercial_pipeline import (
    compute_commercial_views,
    extract_commercial_base_data,
    get_commercial_filter_options,
)
from dashboard.financial_pipeline import (
    DEFAULT_SCHEMA,
    compute_financial_views,
    extract_financial_base_data,
    get_financial_filter_options,
    sanitize_schema,
)
from dashboard.fleet_pipeline import (
    compute_fleet_views,
    extract_fleet_base_data,
    get_fleet_filter_options,
)
from dashboard.hr_pipeline import (
    compute_hr_views,
    extract_hr_base_data,
    get_hr_filter_options,
)


st.set_page_config(page_title="IE Airlines Executive Command Center", layout="wide")


@st.cache_data(ttl=900)
def cached_filter_options(schema_name: str) -> dict[str, object]:
    return get_financial_filter_options(schema=schema_name)


@st.cache_data(ttl=900)
def cached_financial_dataset(
    schema_name: str,
    start_date: object,
    end_date: object,
    classes: tuple[str, ...],
    route_codes: tuple[str, ...],
) -> dict[str, object]:
    base_df = extract_financial_base_data(
        start_date=start_date,
        end_date=end_date,
        classes=classes,
        route_codes=route_codes,
        schema=schema_name,
    )
    return {"base_df": base_df}


@st.cache_data(ttl=900)
def cached_fleet_filter_options(schema_name: str) -> dict[str, object]:
    return get_fleet_filter_options(schema=schema_name)


@st.cache_data(ttl=900)
def cached_fleet_dataset(
    schema_name: str,
    start_date: object,
    end_date: object,
    models: tuple[str, ...],
) -> dict[str, object]:
    base_df = extract_fleet_base_data(
        start_date=start_date,
        end_date=end_date,
        models=models,
        schema=schema_name,
    )
    return {"base_df": base_df}


@st.cache_data(ttl=900)
def cached_commercial_filter_options(schema_name: str) -> dict[str, object]:
    return get_commercial_filter_options(schema=schema_name)


@st.cache_data(ttl=900)
def cached_commercial_dataset(
    schema_name: str,
    start_date: object,
    end_date: object,
    route_codes: tuple[str, ...],
    classes: tuple[str, ...],
    genders: tuple[str, ...],
    countries: tuple[str, ...],
    continents: tuple[str, ...],
) -> dict[str, object]:
    base_df = extract_commercial_base_data(
        start_date=start_date,
        end_date=end_date,
        route_codes=route_codes,
        classes=classes,
        genders=genders,
        countries=countries,
        continents=continents,
        schema=schema_name,
    )
    return {"base_df": base_df}


@st.cache_data(ttl=900)
def cached_hr_filter_options(schema_name: str) -> dict[str, object]:
    return get_hr_filter_options(schema=schema_name)


@st.cache_data(ttl=900)
def cached_hr_dataset(
    schema_name: str,
    start_date: object,
    end_date: object,
    departments: tuple[str, ...],
    jobs: tuple[str, ...],
    genders: tuple[str, ...],
) -> dict[str, object]:
    base_df = extract_hr_base_data(
        start_date=start_date,
        end_date=end_date,
        departments=departments,
        jobs=jobs,
        genders=genders,
        schema=schema_name,
    )
    return {"base_df": base_df}


@st.cache_data(ttl=900)
def cached_overview_payload(
    schema_name: str,
    fuel_price_per_gallon: float,
    flight_window_days: int,
    hr_window_years: int,
) -> dict[str, object]:
    financial_options = cached_filter_options(schema_name)
    fleet_options = cached_fleet_filter_options(schema_name)
    commercial_options = cached_commercial_filter_options(schema_name)
    hr_options = cached_hr_filter_options(schema_name)

    fin_min = financial_options["min_date"]  # type: ignore[index]
    fin_max = financial_options["max_date"]  # type: ignore[index]
    fin_start = max(fin_min, fin_max - timedelta(days=int(flight_window_days)))  # type: ignore[operator]
    financial_base = extract_financial_base_data(
        start_date=fin_start,
        end_date=fin_max,
        classes=tuple(financial_options["classes"]),  # type: ignore[index]
        route_codes=(),
        schema=schema_name,
    )
    financial_views = compute_financial_views(
        base_df=financial_base, fuel_price_per_gallon=float(fuel_price_per_gallon)
    )

    fleet_min = fleet_options["min_date"]  # type: ignore[index]
    fleet_max = fleet_options["max_date"]  # type: ignore[index]
    fleet_start = max(fleet_min, fleet_max - timedelta(days=int(flight_window_days)))  # type: ignore[operator]
    fleet_base = extract_fleet_base_data(
        start_date=fleet_start,
        end_date=fleet_max,
        models=tuple(fleet_options["models"]),  # type: ignore[index]
        schema=schema_name,
    )
    fleet_views = compute_fleet_views(
        base_df=fleet_base,
        maintenance_takeoffs_threshold=3000,
        maintenance_flight_hours_threshold=6000,
        a_check_days_threshold=90,
        b_check_days_threshold=365,
        warning_ratio=0.85,
    )

    comm_min = commercial_options["min_date"]  # type: ignore[index]
    comm_max = commercial_options["max_date"]  # type: ignore[index]
    comm_start = max(comm_min, comm_max - timedelta(days=int(flight_window_days)))  # type: ignore[operator]
    commercial_base = extract_commercial_base_data(
        start_date=comm_start,
        end_date=comm_max,
        route_codes=(),
        classes=tuple(commercial_options["classes"]),  # type: ignore[index]
        genders=tuple(commercial_options["genders"]),  # type: ignore[index]
        countries=(),
        continents=tuple(commercial_options["continents"]),  # type: ignore[index]
        schema=schema_name,
    )
    commercial_views = compute_commercial_views(base_df=commercial_base)

    hr_min = hr_options["min_date"]  # type: ignore[index]
    hr_max = hr_options["max_date"]  # type: ignore[index]
    hr_start = max(hr_min, hr_max - timedelta(days=365 * int(hr_window_years)))  # type: ignore[operator]
    hr_base = extract_hr_base_data(
        start_date=hr_start,
        end_date=hr_max,
        departments=tuple(hr_options["departments"]),  # type: ignore[index]
        jobs=(),
        genders=tuple(hr_options["genders"]),  # type: ignore[index]
        schema=schema_name,
    )
    hr_views = compute_hr_views(base_df=hr_base)

    return {
        "financial_views": financial_views,
        "fleet_views": fleet_views,
        "commercial_views": commercial_views,
        "hr_views": hr_views,
        "date_ranges": {
            "financial_start": fin_start,
            "financial_end": fin_max,
            "fleet_start": fleet_start,
            "fleet_end": fleet_max,
            "commercial_start": comm_start,
            "commercial_end": comm_max,
            "hr_start": hr_start,
            "hr_end": hr_max,
        },
    }


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    return f"{value * 100:,.2f}%"


def render_overview_dashboard(
    payload: dict[str, object],
    fuel_price_per_gallon: float,
    flight_window_days: int,
    hr_window_years: int,
) -> None:
    financial_views = payload["financial_views"]  # type: ignore[index]
    fleet_views = payload["fleet_views"]  # type: ignore[index]
    commercial_views = payload["commercial_views"]  # type: ignore[index]
    hr_views = payload["hr_views"]  # type: ignore[index]
    date_ranges = payload["date_ranges"]  # type: ignore[index]

    fin_kpis = financial_views["kpis"]  # type: ignore[index]
    fleet_kpis = fleet_views["kpis"]  # type: ignore[index]
    comm_kpis = commercial_views["kpis"]  # type: ignore[index]
    hr_kpis = hr_views["kpis"]  # type: ignore[index]

    st.caption(
        f"Executive snapshot based on trailing {int(flight_window_days)} days for flight operations, "
        f"trailing {int(hr_window_years)} years for HR, and fuel assumption ${float(fuel_price_per_gallon):,.2f}/gal."
    )
    st.caption(
        "Data windows: "
        f"Financial {date_ranges['financial_start']} to {date_ranges['financial_end']} | "  # type: ignore[index]
        f"Fleet {date_ranges['fleet_start']} to {date_ranges['fleet_end']} | "  # type: ignore[index]
        f"Commercial {date_ranges['commercial_start']} to {date_ranges['commercial_end']} | "  # type: ignore[index]
        f"HR {date_ranges['hr_start']} to {date_ranges['hr_end']}"  # type: ignore[index]
    )

    kpi_row1 = st.columns(5)
    kpi_row1[0].metric("Revenue", format_currency(float(fin_kpis["total_revenue"])))
    kpi_row1[1].metric("Estimated Profit", format_currency(float(fin_kpis["estimated_profit"])))
    kpi_row1[2].metric("Avg Load Factor", format_percentage(float(comm_kpis["avg_load_factor"])))
    kpi_row1[3].metric("Active Fleet", f"{int(fleet_kpis['active_aircraft']):,}")
    kpi_row1[4].metric("At-Risk Aircraft", f"{int(fleet_kpis['at_risk_aircraft']):,}")

    kpi_row2 = st.columns(5)
    kpi_row2[0].metric("Headcount", f"{int(hr_kpis['total_headcount']):,}")
    kpi_row2[1].metric("Budget Utilization", format_percentage(float(hr_kpis["budget_utilization"])))
    kpi_row2[2].metric("Active Routes", f"{int(comm_kpis['active_routes']):,}")
    kpi_row2[3].metric("Active Countries", f"{int(comm_kpis['active_countries']):,}")
    kpi_row2[4].metric("Ancillary Share", format_percentage(float(fin_kpis["ancillary_share"])))

    recommendations: list[tuple[str, str]] = []
    if float(fin_kpis["estimated_profit"]) < 0:
        recommendations.append(
            (
                "critical",
                "Estimated profitability is negative in the selected window. Trigger immediate route and pricing review.",
            )
        )
    if float(fin_kpis["ancillary_share"]) < 0.15:
        recommendations.append(
            (
                "opportunity",
                "Ancillary share is below 15%. Consider bundled services, seat upsell, and loyalty cross-sell campaigns.",
            )
        )
    if float(comm_kpis["avg_load_factor"]) < 0.72:
        recommendations.append(
            (
                "warning",
                "Average load factor is below target (72%). Optimize frequency on low-demand routes and tighten yield controls.",
            )
        )
    active_aircraft = max(int(fleet_kpis["active_aircraft"]), 1)
    if int(fleet_kpis["at_risk_aircraft"]) / active_aircraft > 0.2:
        recommendations.append(
            (
                "critical",
                "Maintenance risk concentration exceeds 20% of active fleet. Prioritize maintenance slots and spare-aircraft planning.",
            )
        )
    if float(hr_kpis["budget_utilization"]) > 1.0:
        recommendations.append(
            (
                "warning",
                "Workforce compensation is above department budget envelope. Require department-level remediation plans.",
            )
        )
    if float(hr_kpis["crew_to_operational_ratio"]) > 1.25:
        recommendations.append(
            (
                "opportunity",
                "Fleet crew capacity appears high versus operational staffing. Review rostering and utilization strategy.",
            )
        )
    if not recommendations:
        recommendations.append(
            (
                "healthy",
                "No immediate red flags detected. Maintain cadence on route profitability, maintenance, and workforce planning.",
            )
        )

    st.subheader("Executive Recommendations")
    for level, message in recommendations:
        if level == "critical":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        elif level == "opportunity":
            st.info(message)
        else:
            st.success(message)

    route_profit = financial_views["route_profitability"]  # type: ignore[index]
    maintenance_alerts = fleet_views["maintenance_alerts"]  # type: ignore[index]
    country_distribution = commercial_views["country_distribution"]  # type: ignore[index]
    dept_summary = hr_views["department_summary"]  # type: ignore[index]
    daily_revenue = financial_views["daily_revenue"]  # type: ignore[index]

    if not route_profit.empty:
        top_cash = route_profit.sort_values("estimated_profit", ascending=False).head(5).copy()
        top_cash["route"] = (
            top_cash["route_code"].astype(str)
            + " ("
            + top_cash["origin"].astype(str)
            + "-"
            + top_cash["destination"].astype(str)
            + ")"
        )
    else:
        top_cash = pd.DataFrame()

    if not maintenance_alerts.empty:
        maintenance_status = (
            maintenance_alerts.groupby("status", as_index=False)
            .size()
            .rename(columns={"size": "aircraft"})
            .sort_values("aircraft", ascending=False)
        )
    else:
        maintenance_status = pd.DataFrame()

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        if not daily_revenue.empty:
            daily = daily_revenue.copy()
            daily["flight_date"] = pd.to_datetime(daily["flight_date"])
            fig_trend = px.line(
                daily,
                x="flight_date",
                y="total_revenue",
                title="Revenue Trend (Overview Window)",
                labels={"flight_date": "Date", "total_revenue": "Revenue (USD)"},
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        if not maintenance_status.empty:
            fig_maint = px.pie(
                maintenance_status,
                names="status",
                values="aircraft",
                title="Fleet Maintenance Risk Mix",
            )
            st.plotly_chart(fig_maint, use_container_width=True)

    with overview_col2:
        if not top_cash.empty:
            fig_routes = px.bar(
                top_cash,
                x="estimated_profit",
                y="route",
                orientation="h",
                title="Top 5 Cash Cow Routes",
                labels={"estimated_profit": "Estimated Profit (USD)", "route": "Route"},
                color="estimated_profit",
                color_continuous_scale="Tealgrn",
            )
            fig_routes.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_routes, use_container_width=True)
        if not country_distribution.empty:
            fig_country = px.bar(
                country_distribution.head(10),
                x="passenger_country",
                y="passengers",
                title="Top 10 Passenger Countries",
                labels={"passenger_country": "Country", "passengers": "Passengers"},
            )
            fig_country.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_country, use_container_width=True)

    if not dept_summary.empty:
        st.subheader("Top Department Budget Pressure")
        dept = dept_summary.sort_values("budget_utilization", ascending=False).head(10)
        st.dataframe(
            dept[
                [
                    "deptno",
                    "deptname",
                    "headcount",
                    "compensation_total",
                    "dept_budget",
                    "budget_utilization",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_financial_dashboard(
    base_df: pd.DataFrame,
    fuel_price_per_gallon: float,
    top_n_routes: int,
) -> None:
    views = compute_financial_views(base_df=base_df, fuel_price_per_gallon=fuel_price_per_gallon)
    kpis = views["kpis"]
    daily_revenue = views["daily_revenue"]
    route_profitability = views["route_profitability"]
    ancillary_by_class = views["ancillary_by_class"]

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    metric_col1.metric("Total Revenue", format_currency(float(kpis["total_revenue"])))
    metric_col2.metric("RASM", f"${float(kpis['rasm']):,.4f}")
    metric_col3.metric("Estimated Fuel Cost", format_currency(float(kpis["estimated_fuel_cost"])))
    metric_col4.metric("Estimated Total Cost", format_currency(float(kpis["estimated_total_cost"])))
    metric_col5.metric("Estimated Profit", format_currency(float(kpis["estimated_profit"])))
    metric_col6.metric("Ancillary Share", format_percentage(float(kpis["ancillary_share"])))

    st.caption(
        f"Cost model assumption: fuel cost estimated at ${fuel_price_per_gallon:,.2f} per gallon."
    )

    if not daily_revenue.empty:
        trend = daily_revenue.copy()
        trend["flight_date"] = pd.to_datetime(trend["flight_date"])
        trend_long = trend.melt(
            id_vars=["flight_date"],
            value_vars=["total_revenue", "base_revenue", "ancillary_revenue"],
            var_name="revenue_type",
            value_name="revenue",
        )
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            fig_revenue = px.line(
                trend_long,
                x="flight_date",
                y="revenue",
                color="revenue_type",
                title="Revenue Trend",
                labels={
                    "flight_date": "Departure Date",
                    "revenue": "Revenue (USD)",
                    "revenue_type": "Revenue Component",
                },
            )
            fig_revenue.update_layout(legend_title_text="")
            st.plotly_chart(fig_revenue, use_container_width=True)

        with chart_col2:
            avg_daily_revenue = trend["total_revenue"].mean()
            st.metric("Avg Daily Revenue", format_currency(float(avg_daily_revenue)))
            st.metric(
                "Total Available Seat Miles",
                f"{float(kpis['available_seat_miles']):,.0f}",
            )
            if trend["total_revenue"].iloc[0] != 0:
                growth = (
                    (trend["total_revenue"].iloc[-1] - trend["total_revenue"].iloc[0])
                    / trend["total_revenue"].iloc[0]
                )
                st.metric("Revenue Change (Window)", format_percentage(float(growth)))

    if not route_profitability.empty:
        route_data = route_profitability.copy()
        route_data["route_label"] = (
            route_data["route_code"].astype(str)
            + " ("
            + route_data["origin"].astype(str)
            + "-"
            + route_data["destination"].astype(str)
            + ")"
        )

        top_cash_cows = route_data.nlargest(top_n_routes, "estimated_profit")
        top_money_pits = route_data.nsmallest(top_n_routes, "estimated_profit")
        ranked_routes = (
            pd.concat([top_cash_cows, top_money_pits], ignore_index=True)
            .drop_duplicates(subset=["route_code"])
            .sort_values("estimated_profit")
        )

        route_col1, route_col2 = st.columns(2)
        with route_col1:
            fig_routes = px.bar(
                ranked_routes,
                x="estimated_profit",
                y="route_label",
                color="segment",
                orientation="h",
                title='Route Profitability: "Cash Cows" vs "Money Pits"',
                labels={
                    "estimated_profit": "Estimated Profit (USD)",
                    "route_label": "Route",
                    "segment": "Classification",
                },
                color_discrete_map={"Cash Cow": "#2ca02c", "Money Pit": "#d62728"},
            )
            fig_routes.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_routes, use_container_width=True)

        with route_col2:
            fig_scatter = px.scatter(
                route_data,
                x="route_revenue",
                y="estimated_profit",
                size="tickets_sold",
                color="profit_margin",
                color_continuous_scale="RdYlGn",
                hover_name="route_code",
                hover_data={
                    "origin": True,
                    "destination": True,
                    "load_factor": ":.2%",
                    "route_rasm": ":.4f",
                },
                title="Route Revenue vs Profit",
                labels={
                    "route_revenue": "Route Revenue (USD)",
                    "estimated_profit": "Estimated Profit (USD)",
                    "profit_margin": "Profit Margin",
                },
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Route Profitability Detail")
        st.dataframe(
            route_data[
                [
                    "route_code",
                    "origin",
                    "destination",
                    "flights",
                    "tickets_sold",
                    "route_revenue",
                    "estimated_fuel_cost",
                    "estimated_total_cost",
                    "estimated_profit",
                    "profit_margin",
                    "route_rasm",
                    "load_factor",
                    "segment",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not ancillary_by_class.empty:
        ancillary = ancillary_by_class.copy()
        ancillary_col1, ancillary_col2 = st.columns(2)

        with ancillary_col1:
            stacked = ancillary.melt(
                id_vars=["ticket_class"],
                value_vars=["base_revenue", "airport_tax_revenue", "local_tax_revenue"],
                var_name="component",
                value_name="amount",
            )
            fig_ancillary = px.bar(
                stacked,
                x="ticket_class",
                y="amount",
                color="component",
                barmode="stack",
                title="Revenue Composition by Ticket Class",
                labels={
                    "ticket_class": "Ticket Class",
                    "amount": "Revenue (USD)",
                    "component": "Component",
                },
            )
            st.plotly_chart(fig_ancillary, use_container_width=True)

        with ancillary_col2:
            ancillary_totals = {
                "Airport Tax": float(ancillary["airport_tax_revenue"].sum()),
                "Local Tax": float(ancillary["local_tax_revenue"].sum()),
            }
            fig_tax = px.pie(
                names=list(ancillary_totals.keys()),
                values=list(ancillary_totals.values()),
                title="Ancillary Revenue Mix",
            )
            st.plotly_chart(fig_tax, use_container_width=True)

        st.subheader("Ancillary Revenue Detail")
        st.dataframe(
            ancillary[
                [
                    "ticket_class",
                    "base_revenue",
                    "airport_tax_revenue",
                    "local_tax_revenue",
                    "total_revenue",
                    "ancillary_share",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_fleet_dashboard(
    base_df: pd.DataFrame,
    maintenance_takeoffs_threshold: int,
    maintenance_flight_hours_threshold: int,
    a_check_days_threshold: int,
    b_check_days_threshold: int,
    warning_ratio: float,
) -> None:
    views = compute_fleet_views(
        base_df=base_df,
        maintenance_takeoffs_threshold=maintenance_takeoffs_threshold,
        maintenance_flight_hours_threshold=maintenance_flight_hours_threshold,
        a_check_days_threshold=a_check_days_threshold,
        b_check_days_threshold=b_check_days_threshold,
        warning_ratio=warning_ratio,
    )
    kpis = views["kpis"]
    utilization = views["utilization_by_aircraft"]
    daily_ops = views["daily_operations"]
    fuel_efficiency = views["fuel_efficiency_by_model"]
    maintenance_alerts = views["maintenance_alerts"]

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6, metric_col7 = (
        st.columns(7)
    )
    metric_col1.metric("Active Aircraft", f"{int(kpis['active_aircraft']):,}")
    metric_col2.metric("Flights Operated", f"{int(kpis['total_flights']):,}")
    metric_col3.metric("Flight Hours", f"{float(kpis['total_flight_hours']):,.1f}")
    metric_col4.metric(
        "Avg Utilization (hrs/aircraft)",
        f"{float(kpis['avg_utilization_hours_per_aircraft']):,.2f}",
    )
    metric_col5.metric("Route Miles", f"{float(kpis['total_route_distance']):,.0f}")
    metric_col6.metric("Avg Fuel Burn (gal/hr)", f"{float(kpis['avg_fuel_gallons_hour']):,.1f}")
    metric_col7.metric("At-Risk Aircraft", f"{int(kpis['at_risk_aircraft']):,}")

    st.caption(
        "Maintenance risk uses configurable thresholds for A-check age, B-check age, "
        "maintenance takeoffs, and maintenance flight-hours."
    )

    if not daily_ops.empty:
        daily = daily_ops.copy()
        daily["flight_date"] = pd.to_datetime(daily["flight_date"])

        ops_col1, ops_col2 = st.columns(2)
        with ops_col1:
            fig_daily_flights = px.line(
                daily,
                x="flight_date",
                y="flights_operated",
                title="Daily Flights Operated",
                labels={"flight_date": "Date", "flights_operated": "Flights"},
            )
            st.plotly_chart(fig_daily_flights, use_container_width=True)
        with ops_col2:
            fig_daily_hours = px.line(
                daily,
                x="flight_date",
                y="flight_hours_operated",
                title="Daily Flight Hours",
                labels={"flight_date": "Date", "flight_hours_operated": "Flight Hours"},
            )
            st.plotly_chart(fig_daily_hours, use_container_width=True)

    if not utilization.empty:
        utilization_sorted = utilization.sort_values("flight_hours_operated", ascending=False)
        top_utilization = utilization_sorted.head(20).copy()
        top_utilization["aircraft_label"] = (
            top_utilization["aircraft_registration"].astype(str)
            + " ("
            + top_utilization["model"].astype(str)
            + ")"
        )

        util_col1, util_col2 = st.columns(2)
        with util_col1:
            fig_utilization = px.bar(
                top_utilization,
                x="flight_hours_operated",
                y="aircraft_label",
                orientation="h",
                color="avg_fuel_gallons_hour",
                color_continuous_scale="Tealgrn",
                title="Fleet Utilization Leaderboard (Top 20 Aircraft)",
                labels={
                    "flight_hours_operated": "Flight Hours",
                    "aircraft_label": "Aircraft",
                    "avg_fuel_gallons_hour": "Avg Fuel (gal/hr)",
                },
            )
            fig_utilization.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_utilization, use_container_width=True)

        with util_col2:
            fig_fuel_model = px.scatter(
                utilization,
                x="distance_operated",
                y="flight_hours_operated",
                color="model",
                size="avg_fuel_gallons_hour",
                hover_name="aircraft_registration",
                title="Distance vs Flight Hours by Aircraft",
                labels={
                    "distance_operated": "Distance Operated (miles)",
                    "flight_hours_operated": "Flight Hours",
                    "avg_fuel_gallons_hour": "Avg Fuel (gal/hr)",
                },
            )
            st.plotly_chart(fig_fuel_model, use_container_width=True)

        st.subheader("Fleet Utilization Detail")
        st.dataframe(
            utilization[
                [
                    "aircraft_registration",
                    "model",
                    "flights_operated",
                    "flight_hours_operated",
                    "distance_operated",
                    "avg_hours_per_flight",
                    "avg_fuel_gallons_hour",
                    "lifetime_total_flight_distance",
                    "crew_members",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not fuel_efficiency.empty:
        fig_efficiency = px.bar(
            fuel_efficiency,
            x="model",
            y="gallons_per_100_miles",
            color="avg_fuel_gallons_hour",
            color_continuous_scale="Blues",
            title="Fuel Efficiency by Aircraft Model (Lower is Better)",
            labels={
                "model": "Model",
                "gallons_per_100_miles": "Gallons per 100 miles",
                "avg_fuel_gallons_hour": "Avg Fuel (gal/hr)",
            },
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)

        st.subheader("Fuel Efficiency Leaderboard")
        st.dataframe(
            fuel_efficiency[
                [
                    "model",
                    "flights_operated",
                    "avg_fuel_gallons_hour",
                    "avg_route_distance",
                    "avg_flight_hours",
                    "gallons_per_100_miles",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if not maintenance_alerts.empty:
        maintenance = maintenance_alerts.copy()
        maintenance["aircraft_label"] = (
            maintenance["aircraft_registration"].astype(str)
            + " ("
            + maintenance["model"].astype(str)
            + ")"
        )

        fig_maintenance = px.bar(
            maintenance.sort_values("risk_score", ascending=False).head(20),
            x="risk_score",
            y="aircraft_label",
            color="status",
            orientation="h",
            title="Maintenance Health Alerts (Top Risk Aircraft)",
            labels={
                "risk_score": "Risk Score (1.0 = threshold breached)",
                "aircraft_label": "Aircraft",
                "status": "Status",
            },
            color_discrete_map={
                "Overdue": "#d62728",
                "Warning": "#ff7f0e",
                "Healthy": "#2ca02c",
            },
        )
        fig_maintenance.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_maintenance, use_container_width=True)

        st.subheader("Maintenance Alert Table")
        st.dataframe(
            maintenance[
                [
                    "aircraft_registration",
                    "model",
                    "status",
                    "primary_trigger",
                    "risk_score",
                    "maintenance_takeoffs",
                    "maintenance_flight_hours",
                    "days_since_acheck",
                    "days_since_bcheck",
                    "total_flight_distance",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_commercial_dashboard(base_df: pd.DataFrame, top_routes: int) -> None:
    views = compute_commercial_views(base_df=base_df)
    kpis = views["kpis"]
    load_factor = views["load_factor_by_route"]
    heatmap = views["route_heatmap"]
    demographics = views["passenger_demographics"]
    gender_distribution = views["gender_distribution"]
    country_distribution = views["country_distribution"]
    continent_distribution = views["continent_distribution"]

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Passengers", f"{int(kpis['total_passengers']):,}")
    metric_col2.metric("Avg Load Factor", format_percentage(float(kpis["avg_load_factor"])))
    metric_col3.metric("Active Routes", f"{int(kpis['active_routes']):,}")
    metric_col4.metric("Active Countries", f"{int(kpis['active_countries']):,}")
    metric_col5.metric("Busiest Route", str(kpis["busiest_route"]))

    if not load_factor.empty:
        route_df = load_factor.copy().head(top_routes)
        route_df["route_label"] = (
            route_df["route_code"].astype(str)
            + " ("
            + route_df["origin"].astype(str)
            + "-"
            + route_df["destination"].astype(str)
            + ")"
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_load = px.bar(
                route_df.sort_values("avg_load_factor", ascending=False),
                x="route_label",
                y="avg_load_factor",
                color="tickets_sold",
                title="Top Routes by Load Factor",
                labels={
                    "route_label": "Route",
                    "avg_load_factor": "Avg Load Factor",
                    "tickets_sold": "Passengers",
                },
            )
            fig_load.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_load, use_container_width=True)

        with chart_col2:
            fig_route_pax = px.scatter(
                route_df,
                x="tickets_sold",
                y="avg_load_factor",
                color="flights",
                hover_name="route_code",
                title="Route Volume vs Load Factor",
                labels={
                    "tickets_sold": "Passengers",
                    "avg_load_factor": "Avg Load Factor",
                    "flights": "Flights",
                },
            )
            st.plotly_chart(fig_route_pax, use_container_width=True)

    if not heatmap.empty:
        heatmap_df = heatmap.copy().head(top_routes * 5)
        fig_map = go.Figure()
        for _, row in heatmap_df.iterrows():
            fig_map.add_trace(
                go.Scattergeo(
                    lon=[row["origin_longitude"], row["destination_longitude"]],
                    lat=[row["origin_latitude"], row["destination_latitude"]],
                    mode="lines",
                    line={
                        "width": max(1.0, float(row["passengers"]) / max(heatmap_df["passengers"].max(), 1.0) * 6.0),
                        "color": "rgba(40, 120, 181, 0.6)",
                    },
                    opacity=0.8,
                    hoverinfo="text",
                    text=(
                        f"{row['route_code']} ({row['origin']}-{row['destination']})"
                        f"<br>Passengers: {int(row['passengers'])}"
                    ),
                    showlegend=False,
                )
            )
        fig_map.update_layout(
            title="Route Network Heatmap",
            geo={
                "projection_type": "natural earth",
                "showland": True,
                "landcolor": "rgb(245, 245, 245)",
                "countrycolor": "rgb(220, 220, 220)",
            },
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(fig_map, use_container_width=True)

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        if not demographics.empty:
            fig_age = px.bar(
                demographics,
                x="age_bucket",
                y="passengers",
                color="gender",
                barmode="group",
                title="Passenger Demographics by Age & Gender",
                labels={"age_bucket": "Age Group", "passengers": "Passengers", "gender": "Gender"},
            )
            st.plotly_chart(fig_age, use_container_width=True)
    with demo_col2:
        if not gender_distribution.empty:
            fig_gender = px.pie(
                gender_distribution,
                names="gender",
                values="passengers",
                title="Gender Distribution",
            )
            st.plotly_chart(fig_gender, use_container_width=True)

    geo_col1, geo_col2 = st.columns(2)
    with geo_col1:
        if not country_distribution.empty:
            fig_countries = px.bar(
                country_distribution.head(top_routes),
                x="passenger_country",
                y="passengers",
                title="Top Passenger Countries",
                labels={"passenger_country": "Country", "passengers": "Passengers"},
            )
            fig_countries.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_countries, use_container_width=True)
    with geo_col2:
        if not continent_distribution.empty:
            fig_continents = px.pie(
                continent_distribution,
                names="passenger_continent",
                values="passengers",
                title="Passenger Mix by Continent",
            )
            st.plotly_chart(fig_continents, use_container_width=True)


def render_hr_dashboard(base_df: pd.DataFrame, top_departments: int) -> None:
    views = compute_hr_views(base_df=base_df)
    kpis = views["kpis"]
    department_summary = views["department_summary"]
    job_headcount = views["job_headcount"]
    gender_distribution = views["gender_distribution"]
    hire_trend = views["hire_trend"]

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
    metric_col1.metric("Headcount", f"{int(kpis['total_headcount']):,}")
    metric_col2.metric("Total Salary", format_currency(float(kpis["total_salary"])))
    metric_col3.metric("Total Compensation", format_currency(float(kpis["total_compensation"])))
    metric_col4.metric("Dept Budget", format_currency(float(kpis["total_budget"])))
    metric_col5.metric("Budget Utilization", format_percentage(float(kpis["budget_utilization"])))
    metric_col6.metric(
        "Crew / Operational Ratio", f"{float(kpis['crew_to_operational_ratio']):,.2f}"
    )

    st.caption(
        "Staffing efficiency uses total fleet crew capacity from AIRPLANES divided by operational headcount."
    )

    if not department_summary.empty:
        dept = department_summary.copy().head(top_departments)
        dept["dept_label"] = dept["deptno"].astype(str) + " - " + dept["deptname"].astype(str)

        dept_col1, dept_col2 = st.columns(2)
        with dept_col1:
            fig_comp = px.bar(
                dept.sort_values("compensation_total", ascending=False),
                x="dept_label",
                y=["compensation_total", "dept_budget"],
                barmode="group",
                title="Department Compensation vs Budget",
                labels={"value": "USD", "dept_label": "Department", "variable": "Metric"},
            )
            fig_comp.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_comp, use_container_width=True)
        with dept_col2:
            fig_util = px.bar(
                dept.sort_values("budget_utilization", ascending=False),
                x="dept_label",
                y="budget_utilization",
                color="headcount",
                title="Department Budget Utilization",
                labels={
                    "dept_label": "Department",
                    "budget_utilization": "Utilization",
                    "headcount": "Headcount",
                },
            )
            fig_util.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_util, use_container_width=True)

    hr_col1, hr_col2 = st.columns(2)
    with hr_col1:
        if not job_headcount.empty:
            fig_jobs = px.bar(
                job_headcount.head(15),
                x="job",
                y="headcount",
                color="avg_salary",
                title="Top Job Families by Headcount",
                labels={"job": "Job", "headcount": "Headcount", "avg_salary": "Avg Salary"},
            )
            fig_jobs.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_jobs, use_container_width=True)
    with hr_col2:
        if not gender_distribution.empty:
            fig_gender = px.pie(
                gender_distribution,
                names="gender",
                values="headcount",
                title="Workforce Gender Distribution",
            )
            st.plotly_chart(fig_gender, use_container_width=True)

    if not hire_trend.empty:
        fig_hire = px.line(
            hire_trend,
            x="hire_month",
            y="hires",
            title="Hiring Trend",
            labels={"hire_month": "Hire Month", "hires": "Hires"},
        )
        st.plotly_chart(fig_hire, use_container_width=True)

    if not department_summary.empty:
        st.subheader("Department Summary")
        st.dataframe(
            department_summary[
                [
                    "deptno",
                    "deptname",
                    "location",
                    "headcount",
                    "salary_total",
                    "compensation_total",
                    "dept_budget",
                    "budget_utilization",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.title("IE Airlines Executive Command Center")
    st.caption(
        "Executive Overview plus pillar pages for Financial, Fleet, Commercial, and HR are live."
    )

    with st.expander("Execution Plan", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. Overview (Executive): implemented in this version.",
                    "2. Financial Performance: implemented in this version.",
                    "3. Fleet Operations & Efficiency: implemented in this version.",
                    "4. Commercial & Route Network: implemented in this version.",
                    "5. Human Resources: implemented in this version.",
                ]
            )
        )

    pillar = st.sidebar.radio(
        "Dashboard Pillar",
        [
            "Overview",
            "Financial Performance",
            "Fleet Operations & Efficiency",
            "Commercial & Route Network",
            "Human Resources",
        ],
    )

    schema_input = st.sidebar.text_input(
        "Database Schema",
        value=os.getenv("DB_SCHEMA", DEFAULT_SCHEMA),
        help="Default is IEPLANE.",
    )

    try:
        schema_name = sanitize_schema(schema_input)
    except Exception as error:
        st.error(f"Invalid schema value: {error}")
        st.stop()

    if pillar == "Overview":
        flight_window_days = int(
            st.sidebar.slider(
                "Operations Window (Days)",
                min_value=30,
                max_value=365,
                value=120,
                step=15,
                help="Applied to Financial, Fleet, and Commercial snapshot metrics.",
            )
        )
        hr_window_years = int(
            st.sidebar.slider(
                "HR Window (Years)",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
            )
        )
        fuel_price = st.sidebar.number_input(
            "Fuel Price per Gallon (USD)",
            min_value=0.5,
            max_value=20.0,
            value=3.25,
            step=0.25,
        )

        if st.sidebar.button("Refresh Data Cache"):
            st.cache_data.clear()
            st.rerun()

        try:
            payload = cached_overview_payload(
                schema_name=schema_name,
                fuel_price_per_gallon=float(fuel_price),
                flight_window_days=flight_window_days,
                hr_window_years=hr_window_years,
            )
        except Exception as error:
            st.error(f"Unable to load overview data: {error}")
            st.stop()

        render_overview_dashboard(
            payload=payload,
            fuel_price_per_gallon=float(fuel_price),
            flight_window_days=flight_window_days,
            hr_window_years=hr_window_years,
        )
        return

    if pillar == "Financial Performance":
        try:
            options = cached_filter_options(schema_name)
        except Exception as error:
            st.error(f"Unable to load financial filter options: {error}")
            st.stop()

        min_date = options["min_date"]
        max_date = options["max_date"]
        class_options = options["classes"]
        route_options = options["routes"]

        default_start = max(min_date, max_date - timedelta(days=90))
        selected_dates = st.sidebar.date_input(
            "Departure Window",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = selected_dates
            end_date = selected_dates

        selected_classes = st.sidebar.multiselect(
            "Ticket Class",
            options=class_options,
            default=class_options,
        )

        selected_routes = st.sidebar.multiselect(
            "Route Code",
            options=route_options,
            default=[],
            help="Leave empty to include all routes.",
        )

        fuel_price = st.sidebar.number_input(
            "Fuel Price per Gallon (USD)",
            min_value=0.5,
            max_value=20.0,
            value=3.25,
            step=0.25,
        )
        top_n_routes = st.sidebar.slider("Top/Bottom Routes to Show", 3, 20, 8)

        if st.sidebar.button("Refresh Data Cache"):
            st.cache_data.clear()
            st.rerun()

        try:
            payload = cached_financial_dataset(
                schema_name=schema_name,
                start_date=start_date,
                end_date=end_date,
                classes=tuple(selected_classes),
                route_codes=tuple(selected_routes),
            )
        except Exception as error:
            st.error(f"Unable to load financial data: {error}")
            st.stop()

        base_df = payload["base_df"]  # type: ignore[index]
        if base_df.empty:
            st.warning("No financial records found for the selected filters.")
            st.stop()

        render_financial_dashboard(
            base_df=base_df,
            fuel_price_per_gallon=float(fuel_price),
            top_n_routes=int(top_n_routes),
        )
        return

    if pillar == "Fleet Operations & Efficiency":
        try:
            fleet_options = cached_fleet_filter_options(schema_name)
        except Exception as error:
            st.error(f"Unable to load fleet filter options: {error}")
            st.stop()

        min_date = fleet_options["min_date"]
        max_date = fleet_options["max_date"]
        model_options = fleet_options["models"]

        default_start = max(min_date, max_date - timedelta(days=90))
        selected_dates = st.sidebar.date_input(
            "Departure Window",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = selected_dates
            end_date = selected_dates

        selected_models = st.sidebar.multiselect(
            "Aircraft Model",
            options=model_options,
            default=model_options,
        )

        maintenance_takeoffs_threshold = int(
            st.sidebar.number_input(
                "Takeoffs Threshold",
                min_value=100,
                max_value=20000,
                value=3000,
                step=100,
            )
        )
        maintenance_flight_hours_threshold = int(
            st.sidebar.number_input(
                "Maintenance Flight-Hours Threshold",
                min_value=100,
                max_value=20000,
                value=6000,
                step=100,
            )
        )
        a_check_days_threshold = int(
            st.sidebar.number_input(
                "A-Check Days Threshold",
                min_value=30,
                max_value=365,
                value=90,
                step=5,
            )
        )
        b_check_days_threshold = int(
            st.sidebar.number_input(
                "B-Check Days Threshold",
                min_value=90,
                max_value=1095,
                value=365,
                step=15,
            )
        )
        warning_ratio = st.sidebar.slider(
            "Warning Threshold Ratio",
            min_value=0.50,
            max_value=1.00,
            value=0.85,
            step=0.01,
            help="Aircraft become Warning when any maintenance ratio exceeds this value.",
        )

        if st.sidebar.button("Refresh Data Cache"):
            st.cache_data.clear()
            st.rerun()

        try:
            payload = cached_fleet_dataset(
                schema_name=schema_name,
                start_date=start_date,
                end_date=end_date,
                models=tuple(selected_models),
            )
        except Exception as error:
            st.error(f"Unable to load fleet data: {error}")
            st.stop()

        base_df = payload["base_df"]  # type: ignore[index]
        if base_df.empty:
            st.warning("No fleet records found for the selected filters.")
            st.stop()

        render_fleet_dashboard(
            base_df=base_df,
            maintenance_takeoffs_threshold=maintenance_takeoffs_threshold,
            maintenance_flight_hours_threshold=maintenance_flight_hours_threshold,
            a_check_days_threshold=a_check_days_threshold,
            b_check_days_threshold=b_check_days_threshold,
            warning_ratio=float(warning_ratio),
        )
        return

    if pillar == "Commercial & Route Network":
        try:
            commercial_options = cached_commercial_filter_options(schema_name)
        except Exception as error:
            st.error(f"Unable to load commercial filter options: {error}")
            st.stop()

        min_date = commercial_options["min_date"]
        max_date = commercial_options["max_date"]
        route_options = commercial_options["routes"]
        class_options = commercial_options["classes"]
        gender_options = commercial_options["genders"]
        country_options = commercial_options["countries"]
        continent_options = commercial_options["continents"]

        default_start = max(min_date, max_date - timedelta(days=90))
        selected_dates = st.sidebar.date_input(
            "Departure Window",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = selected_dates
            end_date = selected_dates

        selected_routes = st.sidebar.multiselect(
            "Route Code",
            options=route_options,
            default=[],
            help="Leave empty to include all routes.",
        )
        selected_classes = st.sidebar.multiselect(
            "Ticket Class",
            options=class_options,
            default=class_options,
        )
        selected_genders = st.sidebar.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options,
        )
        selected_countries = st.sidebar.multiselect(
            "Passenger Country",
            options=country_options,
            default=[],
            help="Leave empty to include all countries.",
        )
        selected_continents = st.sidebar.multiselect(
            "Passenger Continent",
            options=continent_options,
            default=continent_options,
        )
        top_routes = st.sidebar.slider("Top Routes to Show", 5, 50, 20)

        if st.sidebar.button("Refresh Data Cache"):
            st.cache_data.clear()
            st.rerun()

        try:
            payload = cached_commercial_dataset(
                schema_name=schema_name,
                start_date=start_date,
                end_date=end_date,
                route_codes=tuple(selected_routes),
                classes=tuple(selected_classes),
                genders=tuple(selected_genders),
                countries=tuple(selected_countries),
                continents=tuple(selected_continents),
            )
        except Exception as error:
            st.error(f"Unable to load commercial data: {error}")
            st.stop()

        base_df = payload["base_df"]  # type: ignore[index]
        if base_df.empty:
            st.warning("No commercial records found for the selected filters.")
            st.stop()

        render_commercial_dashboard(base_df=base_df, top_routes=int(top_routes))
        return

    try:
        hr_options = cached_hr_filter_options(schema_name)
    except Exception as error:
        st.error(f"Unable to load HR filter options: {error}")
        st.stop()

    min_date = hr_options["min_date"]
    max_date = hr_options["max_date"]
    dept_options = hr_options["departments"]
    job_options = hr_options["jobs"]
    gender_options = hr_options["genders"]

    default_start = max(min_date, max_date - timedelta(days=365 * 10))
    selected_dates = st.sidebar.date_input(
        "Hire Date Window",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = selected_dates
        end_date = selected_dates

    selected_departments = st.sidebar.multiselect(
        "Department",
        options=dept_options,
        default=dept_options,
    )
    selected_jobs = st.sidebar.multiselect(
        "Job",
        options=job_options,
        default=[],
        help="Leave empty to include all jobs.",
    )
    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
    )
    top_departments = st.sidebar.slider("Top Departments to Show", 5, 30, 12)

    if st.sidebar.button("Refresh Data Cache"):
        st.cache_data.clear()
        st.rerun()

    try:
        payload = cached_hr_dataset(
            schema_name=schema_name,
            start_date=start_date,
            end_date=end_date,
            departments=tuple(selected_departments),
            jobs=tuple(selected_jobs),
            genders=tuple(selected_genders),
        )
    except Exception as error:
        st.error(f"Unable to load HR data: {error}")
        st.stop()

    base_df = payload["base_df"]  # type: ignore[index]
    if base_df.empty:
        st.warning("No HR records found for the selected filters.")
        st.stop()

    render_hr_dashboard(base_df=base_df, top_departments=int(top_departments))


if __name__ == "__main__":
    main()
