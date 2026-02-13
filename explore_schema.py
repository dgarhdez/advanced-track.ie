from database import DatabaseConnector
import pandas as pd

db = DatabaseConnector()

print("--- TICKETS columns ---")
df_tickets = db.get_query_as_df("SELECT * FROM IEMASTER.TICKETS FETCH FIRST 1 ROWS ONLY")
print(df_tickets.columns.tolist())

print("\n--- AIRPLANES columns ---")
df_airplanes = db.get_query_as_df("SELECT * FROM IEPLANE.AIRPLANES FETCH FIRST 1 ROWS ONLY")
print(df_airplanes.columns.tolist())

print("\n--- FLIGHTS columns ---")
df_flights = db.get_query_as_df("SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 1 ROWS ONLY")
print(df_flights.columns.tolist())
