"""ETL Pipeline — Amman Digital Market Customer Analytics

Extracts data from PostgreSQL, transforms it into customer-level summaries,
validates data quality, and loads results to a database table and CSV file.
"""
from sqlalchemy import create_engine
import pandas as pd
import os
from pathlib import Path


def extract(engine):
    with engine.connect() as conn:
        customers = pd.read_sql("SELECT * FROM customers", conn)
        products = pd.read_sql("SELECT * FROM products", conn)
        orders = pd.read_sql("SELECT * FROM orders", conn)
        order_items = pd.read_sql("SELECT * FROM order_items", conn)

        return {
            "customers": customers,
            "products": products,
            "orders": orders,
            "order_items": order_items
        }


def transform(data_dict):
    """Transform raw data into customer-level analytics summary.

    Steps:
    1. Join orders with order_items and products
    2. Compute line_total (quantity * unit_price)
    3. Filter out cancelled orders (status = 'cancelled')
    4. Filter out suspicious quantities (quantity > 100)
    5. Aggregate to customer level: total_orders, total_revenue,
       avg_order_value, top_category

    Args:
        data_dict: dict of DataFrames from extract()

    Returns:
        DataFrame: customer-level summary with columns:
            customer_id, customer_name, city, total_orders,
            total_revenue, avg_order_value, top_category
    """
    customers = data_dict["customers"].copy()
    products = data_dict["products"].copy()
    orders = data_dict["orders"].copy()
    order_items = data_dict["order_items"].copy()

    merged_items = order_items.merge(products, on="product_id", how="left")
    merged_items["line_total"] = merged_items["quantity"] * merged_items["unit_price"]

    active_orders = orders[orders["status"] != "cancelled"]
    filtered_items = merged_items[merged_items["quantity"] <= 100]
    filtered_orders = filtered_items.merge(
        active_orders[["order_id", "customer_id"]], on="order_id", how="inner"
    )

    if filtered_orders.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "customer_name",
                "city",
                "total_orders",
                "total_revenue",
                "avg_order_value",
                "top_category"
            ]
        )

    order_revenue = (
        filtered_orders
        .groupby(["customer_id", "order_id"], as_index=False)
        .agg(order_revenue=("line_total", "sum"))
    )

    customer_orders = (
        order_revenue
        .groupby("customer_id", as_index=False)
        .agg(
            total_orders=("order_id", "nunique"),
            avg_order_value=("order_revenue", "mean")
        )
    )

    customer_revenue = (
        filtered_orders
        .groupby("customer_id", as_index=False)
        .agg(total_revenue=("line_total", "sum"))
    )

    top_category = (
        filtered_orders
        .groupby(["customer_id", "category"], as_index=False)
        .agg(category_revenue=("line_total", "sum"))
        .sort_values(["customer_id", "category_revenue"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .rename(columns={"category": "top_category"})
        [["customer_id", "top_category"]]
    )

    summary = customers[["customer_id", "customer_name", "city"]].merge(
        customer_revenue, on="customer_id", how="inner"
    )
    summary = summary.merge(customer_orders, on="customer_id", how="inner")
    summary = summary.merge(top_category, on="customer_id", how="left")
    summary["avg_order_value"] = summary["avg_order_value"].round(2)

    return summary[
        [
            "customer_id",
            "customer_name",
            "city",
            "total_orders",
            "total_revenue",
            "avg_order_value",
            "top_category"
        ]
    ]


def validate(df):
    """Run data quality checks on the transformed DataFrame.

    Checks:
    - No nulls in customer_id or customer_name
    - total_revenue > 0 for all customers
    - No duplicate customer_ids
    - total_orders > 0 for all customers

    Args:
        df: transformed customer summary DataFrame

    Returns:
        dict: {check_name: bool} for each check

    Raises:
        ValueError: if any critical check fails
    """
    checks = {
        "customer_id_no_null": not df["customer_id"].isnull().any(),
        "customer_name_no_null": not df["customer_name"].isnull().any(),
        "total_revenue_positive": (df["total_revenue"] > 0).all(),
        "no_duplicate_customer_id": not df["customer_id"].duplicated().any(),
        "total_orders_positive": (df["total_orders"] > 0).all()
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Validation failed: {', '.join(failed)}")

    return checks


def load(df, engine, csv_path):
    """Load customer summary to PostgreSQL table and CSV file.

    Args:
        df: validated customer summary DataFrame
        engine: SQLAlchemy engine
        csv_path: path for CSV output
    """
    output_file = Path(csv_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_sql("customer_summary", con=engine, if_exists="replace", index=False)
    df.to_csv(output_file, index=False)


def main():
    """Orchestrate the ETL pipeline: extract -> transform -> validate -> load."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost/amman_market"
    )
    engine = create_engine(database_url)

    data_dict = extract(engine)
    print("Extracted data frames:", {k: len(v) for k, v in data_dict.items()})

    summary = transform(data_dict)
    validation_results = validate(summary)

    load(summary, engine, Path("output") / "customer_analytics.csv")
    print("ETL complete. Validation results:", validation_results)


if __name__ == "__main__":
    main()
