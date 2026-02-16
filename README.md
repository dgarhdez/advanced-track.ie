# advanced-track.ie

IE's MBDS program for advanced track

## Getting Started: Fork & Clone

Each group must work on their own copy of this repository. Follow these steps to set up your group's repo.

### 1. Fork the Repository

1. Go to the original repository on GitHub.
2. Click the **Fork** button (top-right corner).
3. Select one group member's GitHub account as the destination. This creates a copy of the repo under that account.

### 2. Clone Your Fork

Once forked, every group member should clone the fork to their local machine.

**macOS (Terminal):**
```bash
git clone https://github.com/<your-github-username>/advanced-track.ie.git
cd advanced-track.ie
```

**Windows (PowerShell or Git Bash):**
```powershell
git clone https://github.com/<your-github-username>/advanced-track.ie.git
cd advanced-track.ie
```

> Replace `<your-github-username>` with the GitHub username of the group member who forked the repo.

### 3. Add Group Members as Collaborators

The group member who owns the fork should add the rest of the group:

1. Go to your forked repo on GitHub.
2. Navigate to **Settings > Collaborators**.
3. Click **Add people** and invite each group member by their GitHub username or email.

Once added, all members can push and pull from the same fork.

---

## Database Connection Guide

This guide explains how to set up your Python environment and connect to the IBM DB2 database used in this project.

## 1. Required Libraries

The project relies on a specific set of Python libraries defined in `pyproject.toml`.

### Core Data & Analysis

* **`pandas`**: For data manipulation and analysis. Used to load query results into DataFrames.
* **`numpy`**: Fundamental package for scientific computing.
* **`scikit-learn`**: Machine learning library.
* **`plotly`**: For interactive graphing and visualization.

### Database Connectivity

* **`sqlalchemy`**: The Python SQL Toolkit and Object Relational Mapper. It provides the core interface for connecting to the database.
* **`ibm-db-sa`**: The SQLAlchemy adapter for IBM DB2. This allows SQLAlchemy to communicate specifically with DB2 databases.
  * *Note: This automatically installs the low-level `ibm_db` driver.*
* **`pyodbc`**: A standard ODBC driver (included for compatibility/alternative connection methods).
* **`duckdb`**: An in-process SQL OLAP database management system. Used here to perform fast analytical queries on local Pandas DataFrames.

### Utilities

* **`python-dotenv`**: For managing environment variables (e.g., keeping credentials secure).
* **`azure-identity`**: For Azure authentication support.

## 2. Environment Setup

This project uses `uv` for dependency management.

### Step 1: Install uv

If you don't have `uv` installed, follow the instructions in the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

**macOS / Linux (Terminal):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Install Dependencies

Run the following command in your terminal to create the virtual environment and install all required packages:

```bash
uv sync
```

### Step 3: Activate the Virtual Environment

**macOS / Linux (Terminal):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

### Step 4: Create Your `.env` File

Create a file called `.env` in the project root to store your database credentials securely. This file is already in `.gitignore` so it will not be pushed to GitHub.

**macOS / Linux (Terminal):**
```bash
touch .env
```

**Windows (PowerShell):**
```powershell
New-Item .env
```

Then open `.env` in any text editor and add:

```
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_HOST=52.211.123.34
DB_PORT=25010
DB_NAME=IEMASTER
```

## 3. Connecting to the Database

We use **SQLAlchemy** to establish the connection. The connection string follows the standard URL format for the `ibm_db_sa` dialect.

### Connection String Format

```
db2+ibm_db://<username>:<password>@<host>:<port>/<database>
```

### Python Code Example

Here is how to connect and run a query, based on `test_connection.ipynb`.

#### Step 1: Create the Engine

```python
from sqlalchemy import create_engine

# Replace with your actual credentials
connection_string = "db2+ibm_db://<username>:<password>@52.211.123.34:25010/IEMASTER"

engine = create_engine(connection_string)
```

#### Step 2: Querying with Pandas

The most efficient way to retrieve data for analysis is using `pandas.read_sql`.

```python
import pandas as pd

query = "SELECT * FROM IEPLANE.FLIGHTS FETCH FIRST 10 ROWS ONLY"
df = pd.read_sql(query, engine)

print(df.head())
```

## ATT Project Instructions

Now you have access to the data through pandas, you should follow these steps.

1. Create a python class to connect to the database and fetch data.

    * In order to connect, create an `env` file to store your credentials securely and use `python-dotenv` to load them into your connector class.
    * This class should have methods to execute queries and return data as pandas DataFrames.
    * Include tests to verify the connection and data retrieval.

2. With the connection ready, use streamlit to build an interactive dashboard.

    * The dashboard should cover the key pillars and KPIs outlined in the project proposal (PROJECT_PROPOSAL.md).
    * Use plotly for visualizations.
    * Implement filters and interactive elements to explore the data.

3. Deliver your project by:

    * Forking this repository (see [Getting Started](#getting-started-fork--clone) above).
    * Committing your code and pushing it to your fork.
    * Sharing the link to your forked repository with all necessary instructions to run your project.

## 4. Run the Executive Command Center

This repository now includes a Streamlit app entry point at `streamlit_app.py`.

### Current scope (Phases 1-2)

The current implementation delivers:

**Financial Performance**

* Total Revenue (`TICKETS.total_amount`)
* RASM (`Total Revenue / Available Seat Miles`)
* Route profitability with "Cash Cow" vs "Money Pit" classification
* Ancillary revenue analysis (`airport_tax` and `local_tax` versus base `price`)
* Estimated operating cost including:
  * Fuel cost (`AIRPLANES.fuel_gallons_hour` * flight duration * fuel price)
  * Staff cost from federated `STAFF` + `FLIGHT_CREW` tables (pilot/attendant average salary model)

**Fleet Operations & Efficiency**

* Fleet utilization by aircraft and model
* Maintenance health alerts (A-check, B-check, takeoffs, flight-hours thresholds)
* Fuel efficiency leaderboard by model

### Run locally

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Optional environment variable (defaults to `IEPLANE`):

```bash
DB_SCHEMA=IEPLANE
```

## 5. Troubleshooting

* **`NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:db2.ibm_db`**:
    This means the `ibm-db-sa` library is missing. Ensure it is in your `pyproject.toml` and you have run `uv sync`.

* **Connection Timeouts**:
    Ensure you are connected to the internet and that the firewall allows traffic to port `25010`.

* **`uv` not recognized (Windows)**:
    Close and reopen your terminal after installing `uv` so the PATH updates take effect.

* **PowerShell script execution disabled (Windows)**:
    If you get an error activating the virtual environment, run this first:
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```
