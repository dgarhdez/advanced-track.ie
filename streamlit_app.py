from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.financial_pipeline import (
    DEFAULT_ATTENDANT_ANNUAL_HOURS,
    DEFAULT_PILOT_ANNUAL_HOURS,
    DEFAULT_SCHEMA,
    compute_financial_views,
    enrich_financial_base_with_staff_costs,
    extract_financial_base_data,
    get_financial_filter_options,
    sanitize_schema,
)
from dashboard.fleet_pipeline import (
    compute_fleet_views,
    extract_fleet_base_data,
    get_fleet_filter_options,
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
def cached_financial_dataset_with_staff_costs(
    schema_name: str,
    start_date: object,
    end_date: object,
    classes: tuple[str, ...],
    route_codes: tuple[str, ...],
    pilot_annual_hours: float,
    attendant_annual_hours: float,
) -> dict[str, object]:
    base_df = extract_financial_base_data(
        start_date=start_date,
        end_date=end_date,
        classes=classes,
        route_codes=route_codes,
        schema=schema_name,
    )
    enriched_df, labor_metadata = enrich_financial_base_with_staff_costs(
        base_df=base_df,
        start_date=start_date,
        end_date=end_date,
        schema=schema_name,
        pilot_annual_hours=pilot_annual_hours,
        attendant_annual_hours=attendant_annual_hours,
    )
    return {"base_df": enriched_df, "labor_metadata": labor_metadata}


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


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    return f"{value * 100:,.2f}%"


def render_financial_dashboard(
    base_df: pd.DataFrame,
    fuel_price_per_gallon: float,
    top_n_routes: int,
    labor_metadata: dict[str, object] | None = None,
) -> None:
    views = compute_financial_views(base_df=base_df, fuel_price_per_gallon=fuel_price_per_gallon)
    kpis = views["kpis"]
    daily_revenue = views["daily_revenue"]
    route_profitability = views["route_profitability"]
    ancillary_by_class = views["ancillary_by_class"]

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6, metric_col7 = (
        st.columns(7)
    )
    metric_col1.metric("Total Revenue", format_currency(float(kpis["total_revenue"])))
    metric_col2.metric("RASM", f"${float(kpis['rasm']):,.4f}")
    metric_col3.metric("Estimated Fuel Cost", format_currency(float(kpis["estimated_fuel_cost"])))
    metric_col4.metric("Estimated Staff Cost", format_currency(float(kpis["estimated_staff_cost"])))
    metric_col5.metric("Estimated Total Cost", format_currency(float(kpis["estimated_total_cost"])))
    metric_col6.metric("Estimated Profit", format_currency(float(kpis["estimated_profit"])))
    metric_col7.metric("Ancillary Share", format_percentage(float(kpis["ancillary_share"])))

    st.caption(
        f"Cost model assumption: fuel cost estimated at ${fuel_price_per_gallon:,.2f} per gallon."
    )
    if labor_metadata:
        if labor_metadata.get("staff_costs_applied"):
            st.caption(
                "Labor model: avg pilot salary "
                f"{format_currency(float(labor_metadata.get('avg_pilot_salary', 0.0)))} and "
                "avg flight attendant salary "
                f"{format_currency(float(labor_metadata.get('avg_attendant_salary', 0.0)))} "
                f"converted to hourly using {float(labor_metadata.get('pilot_annual_hours', 0.0)):,.0f} "
                "pilot annual flight-hours and "
                f"{float(labor_metadata.get('attendant_annual_hours', 0.0)):,.0f} attendant annual "
                "flight-hours."
            )
        else:
            st.caption(f"Labor model status: {labor_metadata.get('message', 'Not available.')}")

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
                    "pilot_assignments",
                    "attendant_assignments",
                    "route_revenue",
                    "estimated_fuel_cost",
                    "estimated_staff_cost",
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


def main() -> None:
    st.title("IE Airlines Executive Command Center")
    st.caption("Pillar-by-pillar buildout. Financial Performance and Fleet Ops are live.")

    with st.expander("Execution Plan", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. Financial Performance: implemented in this version.",
                    "2. Fleet Operations & Efficiency: implemented in this version.",
                    "3. Commercial & Route Network: next implementation phase.",
                    "4. Human Resources: final pillar with staffing KPIs.",
                ]
            )
        )

    pillar = st.sidebar.radio(
        "Dashboard Pillar",
        [
            "Financial Performance",
            "Fleet Operations & Efficiency",
            "Commercial & Route Network (Planned)",
            "Human Resources (Planned)",
        ],
    )

    if pillar in {"Commercial & Route Network (Planned)", "Human Resources (Planned)"}:
        st.info("This page is queued for the next pillar iteration.")
        st.stop()

    schema_input = st.sidebar.text_input(
        "Database Schema",
        value=os.getenv("DB_SCHEMA", DEFAULT_SCHEMA),
        help="Default is IEPLANE.",
    )

    try:
        schema_name = sanitize_schema(schema_input)
        options = cached_filter_options(schema_name)
    except Exception as error:
        st.error(f"Unable to load dashboard filter options: {error}")
        st.stop()

    if pillar == "Financial Performance":
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
        include_staff_costs = st.sidebar.checkbox(
            "Include Federated Staff Costs (STAFF + FLIGHT_CREW)",
            value=True,
        )
        pilot_annual_hours = st.sidebar.number_input(
            "Pilot Annual Flight-Hours",
            min_value=100.0,
            max_value=2000.0,
            value=float(DEFAULT_PILOT_ANNUAL_HOURS),
            step=50.0,
            disabled=not include_staff_costs,
        )
        attendant_annual_hours = st.sidebar.number_input(
            "Flight Attendant Annual Flight-Hours",
            min_value=100.0,
            max_value=2500.0,
            value=float(DEFAULT_ATTENDANT_ANNUAL_HOURS),
            step=50.0,
            disabled=not include_staff_costs,
        )
        top_n_routes = st.sidebar.slider("Top/Bottom Routes to Show", 3, 20, 8)

        if st.sidebar.button("Refresh Data Cache"):
            st.cache_data.clear()
            st.rerun()

        try:
            if include_staff_costs:
                payload = cached_financial_dataset_with_staff_costs(
                    schema_name=schema_name,
                    start_date=start_date,
                    end_date=end_date,
                    classes=tuple(selected_classes),
                    route_codes=tuple(selected_routes),
                    pilot_annual_hours=float(pilot_annual_hours),
                    attendant_annual_hours=float(attendant_annual_hours),
                )
            else:
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
        labor_metadata = payload.get("labor_metadata") if isinstance(payload, dict) else None
        if not include_staff_costs:
            labor_metadata = {
                "staff_costs_applied": False,
                "message": "Disabled from sidebar filter.",
            }

        if base_df.empty:
            st.warning("No financial records found for the selected filters.")
            st.stop()

        render_financial_dashboard(
            base_df=base_df,
            fuel_price_per_gallon=float(fuel_price),
            top_n_routes=int(top_n_routes),
            labor_metadata=labor_metadata if isinstance(labor_metadata, dict) else None,
        )
        return

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


if __name__ == "__main__":
    main()
