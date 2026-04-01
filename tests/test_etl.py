"""Tests for the ETL pipeline.

Write at least 3 tests:
1. test_transform_filters_cancelled — cancelled orders excluded after transform
2. test_transform_filters_suspicious_quantity — quantities > 100 excluded
3. test_validate_catches_nulls — validate() raises ValueError on null customer_id
"""
import pandas as pd
import pytest

from etl_pipeline import transform, validate


def test_transform_filters_cancelled():
    """Create test DataFrames with a cancelled order. Confirm it's excluded."""
    customers = pd.DataFrame(
        [{"customer_id": 1, "customer_name": "Alice", "city": "Amman"}]
    )
    products = pd.DataFrame(
        [{"product_id": 10, "category": "Grocery", "unit_price": 5.0}]
    )
    orders = pd.DataFrame(
        [
            {"order_id": 100, "customer_id": 1, "status": "completed"},
            {"order_id": 101, "customer_id": 1, "status": "cancelled"}
        ]
    )
    order_items = pd.DataFrame(
        [
            {"order_id": 100, "product_id": 10, "quantity": 1},
            {"order_id": 101, "product_id": 10, "quantity": 1}
        ]
    )

    result = transform(
        {
            "customers": customers,
            "products": products,
            "orders": orders,
            "order_items": order_items
        }
    )

    assert len(result) == 1
    assert result.iloc[0]["total_orders"] == 1
    assert result.iloc[0]["total_revenue"] == 5.0


def test_transform_filters_suspicious_quantity():
    """Create test DataFrames with quantity > 100. Confirm it's excluded."""
    customers = pd.DataFrame(
        [{"customer_id": 1, "customer_name": "Bob", "city": "Irbid"}]
    )
    products = pd.DataFrame(
        [{"product_id": 10, "category": "Electronics", "unit_price": 10.0}]
    )
    orders = pd.DataFrame(
        [{"order_id": 100, "customer_id": 1, "status": "completed"}]
    )
    order_items = pd.DataFrame(
        [{"order_id": 100, "product_id": 10, "quantity": 150}]
    )

    result = transform(
        {
            "customers": customers,
            "products": products,
            "orders": orders,
            "order_items": order_items
        }
    )

    assert result.empty


def test_validate_catches_nulls():
    """Create a DataFrame with null customer_id. Confirm validate() raises ValueError."""
    df = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "customer_name": "Alice",
                "city": "Amman",
                "total_orders": 1,
                "total_revenue": 10.0,
                "avg_order_value": 10.0,
                "top_category": "Grocery"
            },
            {
                "customer_id": None,
                "customer_name": "Bob",
                "city": "Irbid",
                "total_orders": 1,
                "total_revenue": 5.0,
                "avg_order_value": 5.0,
                "top_category": "Grocery"
            }
        ]
    )

    with pytest.raises(ValueError, match="customer_id"):
        validate(df)
