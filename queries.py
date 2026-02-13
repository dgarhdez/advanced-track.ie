# Flight related queries
GET_FLIGHTS = "SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 100 ROWS ONLY"

# Plane related queries
GET_AIRPLANES = "SELECT * FROM IEPLANE.AIRPLANES"

# Ticket related queries
GET_TICKETS = "SELECT * FROM IEPLANE.TICKETS FETCH FIRST 100 ROWS ONLY"

# KPI Queries
GET_TOTAL_REVENUE = "SELECT SUM(total_amount) as total_revenue FROM IEPLANE.TICKETS"

GET_ACTIVE_FLEET_COUNT = "SELECT COUNT(*) as fleet_count FROM IEPLANE.AIRPLANES"

# Complex: Average Load Factor
# Calculates: (Sold Seats / Total Capacity) per flight, then averages them.
GET_AVG_LOAD_FACTOR = """
WITH flight_occupancy AS (
    SELECT 
        f.flight_id, 
        COUNT(t.ticket_id) as sold_seats,
        (a.seats_business + a.seats_premium + a.seats_economy) as total_capacity
    FROM IEPLANE.FLIGHTS f
    JOIN IEPLANE.AIRPLANES a ON f.airplane = a.aircraft_registration
    LEFT JOIN IEPLANE.TICKETS t ON f.flight_id = t.flight_id
    GROUP BY f.flight_id, a.seats_business, a.seats_premium, a.seats_economy
)
SELECT AVG(CAST(sold_seats AS DOUBLE) / total_capacity) * 100 as avg_load_factor
FROM flight_occupancy
WHERE total_capacity > 0
"""

# Route Network Queries
GET_ROUTE_NETWORK = """
SELECT 
    r.route_code,
    r.origin,
    r.destination,
    r.distance,
    a1.latitude as origin_lat,
    a1.longitude as origin_lon,
    a2.latitude as dest_lat,
    a2.longitude as dest_lon,
    COUNT(f.flight_id) as flight_count
FROM IEPLANE.ROUTES r
JOIN IEPLANE.AIRPORTS a1 ON r.origin = a1.iata_code
JOIN IEPLANE.AIRPORTS a2 ON r.destination = a2.iata_code
LEFT JOIN IEPLANE.FLIGHTS f ON r.route_code = f.route_code
GROUP BY r.route_code, r.origin, r.destination, r.distance, a1.latitude, a1.longitude, a2.latitude, a2.longitude
"""

# Financial Queries
GET_REVENUE_BY_CLASS = """
SELECT class, SUM(total_amount) as revenue
FROM IEPLANE.TICKETS
GROUP BY class
"""

GET_ROUTE_PROFITABILITY = """
WITH route_stats AS (
    SELECT 
        r.route_code,
        r.origin,
        r.destination,
        SUM(t.total_amount) as total_revenue,
        AVG(a.fuel_gallons_hour * (CAST(r.flight_minutes AS DOUBLE) / 60) * 5.0) as estimated_fuel_cost, -- Assuming $5 per gallon
        COUNT(DISTINCT f.flight_id) as flight_count
    FROM IEPLANE.ROUTES r
    JOIN IEPLANE.FLIGHTS f ON r.route_code = f.route_code
    JOIN IEPLANE.AIRPLANES a ON f.airplane = a.aircraft_registration
    LEFT JOIN IEPLANE.TICKETS t ON f.flight_id = t.flight_id
    GROUP BY r.route_code, r.origin, r.destination
)
SELECT 
    *,
    (total_revenue - (estimated_fuel_cost * flight_count)) as estimated_profit
FROM route_stats
ORDER BY estimated_profit DESC
"""

# Metric Components
GET_TOTAL_ASM = """
SELECT SUM((a.seats_business + a.seats_premium + a.seats_economy) * r.distance) as total_asm
FROM IEPLANE.FLIGHTS f
JOIN IEPLANE.AIRPLANES a ON f.airplane = a.aircraft_registration
JOIN IEPLANE.ROUTES r ON f.route_code = r.route_code
"""
