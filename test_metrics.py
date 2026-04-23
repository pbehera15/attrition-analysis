import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "monthly_income": [4000, 6000, 5000, 7000, 4500, 8000],
            "job_satisfaction": [1, 2, 2, 4, 1, 4],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 2
    assert sales["attrition_rate"] == 100.0

    hr = result[result["department"] == "HR"].iloc[0]
    assert hr["employees"] == 2
    assert hr["leavers"] == 1
    assert hr["attrition_rate"] == 50.0

    it = result[result["department"] == "IT"].iloc[0]
    assert it["leavers"] == 0
    assert it["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df)
    # overtime=Yes: employees 1,2,5 → leavers 1,2 → rate 66.67%
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    assert yes_row["employees"] == 3
    assert yes_row["leavers"] == 2
    assert yes_row["attrition_rate"] == 66.67

    # overtime=No: employees 3,4,6 → leavers 3 → rate 33.33%
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert no_row["employees"] == 3
    assert no_row["leavers"] == 1
    assert no_row["attrition_rate"] == 33.33


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    # Leavers: 4000, 6000, 5000 → avg 5000.0
    leavers = result[result["attrition"] == "Yes"].iloc[0]
    assert leavers["avg_monthly_income"] == 5000.0

    # Stayers: 7000, 4500, 8000 → avg 6500.0
    stayers = result[result["attrition"] == "No"].iloc[0]
    assert stayers["avg_monthly_income"] == 6500.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_per_total_leavers():
    # Two groups with equal leavers/employees ratios should have equal rates.
    # The old bug would give different rates because it divided by total leavers.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "attrition": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "job_satisfaction": [2, 2, 2, 2, 4, 4],
        }
    )
    result = satisfaction_summary(df)
    # Group 2: 4 employees, 2 leavers → 50.0%
    group2 = result[result["job_satisfaction"] == 2].iloc[0]
    assert group2["attrition_rate"] == 50.0

    # Group 4: 2 employees, 1 leaver → 50.0%
    group4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert group4["attrition_rate"] == 50.0


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
