from database import DatabaseConnector
import pandas as pd

db = DatabaseConnector()

print("--- ROUTES columns ---")
df_routes = db.get_query_as_df("SELECT * FROM IEPLANE.ROUTES FETCH FIRST 1 ROWS ONLY")
print(df_routes.columns.tolist())

print("\n--- AIRPORTS columns ---")
df_airports = db.get_query_as_df("SELECT * FROM IEPLANE.AIRPORTS FETCH FIRST 1 ROWS ONLY")
print(df_airports.columns.tolist())
