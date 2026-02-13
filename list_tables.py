from database import DatabaseConnector
import pandas as pd

db = DatabaseConnector()

# Query to list all tables accessible to the user
query = "SELECT TABSCHEMA, TABNAME FROM SYSCAT.TABLES WHERE TABSCHEMA NOT LIKE 'SYS%'"
try:
    df_tables = db.get_query_as_df(query)
    print("Available tables:")
    print(df_tables)
except Exception as e:
    print(f"Error listing tables: {e}")
