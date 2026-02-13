from database import DatabaseConnector
import queries
import pandas as pd

# Initialize connection
print("Initializing DatabaseConnector...")
try:
    db = DatabaseConnector()
    print("DatabaseConnector initialized.")
except Exception as e:
    print(f"Error initializing DatabaseConnector: {e}")
    exit(1)

# Debug: Check FLIGHTS table count
print("\nDebugging FLIGHTS table...")
try:
    count_query = "SELECT COUNT(*) as count FROM IEPLANE.FLIGHTS"
    count_df = db.get_query_as_df(count_query)
    print("FLIGHTS table count:")
    print(count_df)
except Exception as e:
    print(f"Error executing count query: {e}")

# Debug: Run the exact query used in app
print(f"\nDebugging query: {queries.GET_FLIGHTS}")
try:
    flights_df = db.get_query_as_df(queries.GET_FLIGHTS)
    print(f"DataFrame shape: {flights_df.shape}")
    if flights_df.empty:
        print("DataFrame is empty.")
    else:
        print("DataFrame head:")
        print(flights_df.head())
except Exception as e:
    print(f"Error executing flights query: {e}")
