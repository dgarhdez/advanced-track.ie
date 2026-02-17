from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from dashboard.commercial_pipeline import (
    build_commercial_filters,
    compute_commercial_views,
)


class TestCommercialPipeline(unittest.TestCase):
    def test_build_commercial_filters_with_all_dimensions(self) -> None:
        where_clause, params = build_commercial_filters(
            alias="t",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            route_codes=["R001"],
            classes=["E", "B"],
            genders=["F"],
            countries=["United States"],
            continents=["North America"],
        )

        self.assertIn("t.departure >= :start_departure", where_clause)
        self.assertIn("t.departure < :end_departure", where_clause)
        self.assertIn("t.route_code IN (:route_0)", where_clause)
        self.assertIn("t.class IN (:class_0, :class_1)", where_clause)
        self.assertIn("p.gender IN (:gender_0)", where_clause)
        self.assertIn("COALESCE(c.name, p.country) IN (:country_0)", where_clause)
        self.assertIn("COALESCE(c.continent, 'Unknown') IN (:continent_0)", where_clause)
        self.assertEqual(params["route_0"], "R001")
        self.assertEqual(params["class_0"], "E")
        self.assertEqual(params["class_1"], "B")
        self.assertEqual(params["gender_0"], "F")
        self.assertEqual(params["country_0"], "United States")
        self.assertEqual(params["continent_0"], "North America")

    def test_compute_commercial_views(self) -> None:
        base_df = pd.DataFrame(
            [
                {
                    "ticket_id": "T1",
                    "passenger_id": 1,
                    "flight_id": "IE1001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "ticket_class": "E",
                    "gender": "F",
                    "birth_date": "1990-01-01",
                    "passenger_country": "United States",
                    "passenger_continent": "North America",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "total_seats": 2,
                    "origin_latitude": 40.6413,
                    "origin_longitude": -73.7781,
                    "destination_latitude": 33.9416,
                    "destination_longitude": -118.4085,
                },
                {
                    "ticket_id": "T2",
                    "passenger_id": 2,
                    "flight_id": "IE1001",
                    "route_code": "R001",
                    "departure": "2025-01-10 08:00:00",
                    "ticket_class": "E",
                    "gender": "M",
                    "birth_date": "1985-01-01",
                    "passenger_country": "United States",
                    "passenger_continent": "North America",
                    "origin": "JFK",
                    "destination": "LAX",
                    "distance": 500,
                    "total_seats": 2,
                    "origin_latitude": 40.6413,
                    "origin_longitude": -73.7781,
                    "destination_latitude": 33.9416,
                    "destination_longitude": -118.4085,
                },
                {
                    "ticket_id": "T3",
                    "passenger_id": 3,
                    "flight_id": "IE2001",
                    "route_code": "R002",
                    "departure": "2025-01-11 09:00:00",
                    "ticket_class": "B",
                    "gender": "F",
                    "birth_date": "2004-01-01",
                    "passenger_country": "United Kingdom",
                    "passenger_continent": "Europe",
                    "origin": "LHR",
                    "destination": "CDG",
                    "distance": 300,
                    "total_seats": 3,
                    "origin_latitude": 51.47,
                    "origin_longitude": -0.4543,
                    "destination_latitude": 49.0097,
                    "destination_longitude": 2.5479,
                },
            ]
        )

        views = compute_commercial_views(base_df)
        kpis = views["kpis"]
        load_factor = views["load_factor_by_route"]
        heatmap = views["route_heatmap"]
        demographics = views["passenger_demographics"]
        country_distribution = views["country_distribution"]
        continent_distribution = views["continent_distribution"]

        self.assertEqual(int(kpis["total_passengers"]), 3)
        self.assertEqual(int(kpis["active_routes"]), 2)
        self.assertEqual(int(kpis["active_countries"]), 2)
        self.assertEqual(str(kpis["busiest_route"]), "R001")
        self.assertAlmostEqual(float(kpis["avg_load_factor"]), (1.0 + (1.0 / 3.0)) / 2.0, places=6)

        r001 = load_factor.loc[load_factor["route_code"] == "R001"].iloc[0]
        self.assertEqual(int(r001["tickets_sold"]), 2)
        self.assertAlmostEqual(float(r001["avg_load_factor"]), 1.0, places=6)
        self.assertEqual(len(heatmap), 2)
        self.assertGreater(len(demographics), 0)
        self.assertEqual(int(country_distribution.iloc[0]["passengers"]), 2)
        self.assertEqual(int(continent_distribution["passengers"].sum()), 3)


if __name__ == "__main__":
    unittest.main()
