import unittest
import pandas as pd
from tools import (
    dataset_overview, column_profile, missing_data_report, value_counts,
    grouped_aggregation, filter_rows, correlation_analysis,
)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "region": ["North", "North", "South", "South"],
            "revenue": [100.0, 200.0, 80.0, 120.0],
            "profit": [30.0, 70.0, 20.0, 40.0],
            "status": ["Done", "Done", "Pending", None],
        })

    def test_dataset_overview(self):
        result = dataset_overview(self.df)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["columns_count"], 4)

    def test_column_profile_numeric(self):
        result = column_profile(self.df, "revenue")
        self.assertEqual(result["kind"], "numeric")
        self.assertEqual(result["statistics"]["mean"], 125.0)

    def test_missing_report(self):
        result = missing_data_report(self.df)
        self.assertEqual(result["columns_ranked_by_missing"][0]["column"], "status")

    def test_value_counts(self):
        result = value_counts(self.df, "region")
        self.assertEqual(result["frequencies"][0]["count"], 2)

    def test_group_aggregation(self):
        result = grouped_aggregation(self.df, "region", "revenue", "sum")
        self.assertEqual(result["results"][0]["group"], "North")
        self.assertEqual(result["results"][0]["value"], 300.0)

    def test_filter_rows(self):
        result = filter_rows(self.df, "revenue", ">", "100")
        self.assertEqual(result["matching_rows"], 2)

    def test_correlations(self):
        result = correlation_analysis(self.df)
        self.assertIn("pairs", result)
        self.assertTrue(result["pairs"])

    def test_unknown_column_is_safe_error(self):
        with self.assertRaises(ValueError):
            column_profile(self.df, "missing_name")


if __name__ == "__main__":
    unittest.main()
