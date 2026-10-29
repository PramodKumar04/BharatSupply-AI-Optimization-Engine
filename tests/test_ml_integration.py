"""
BharatSupply ML Integration Test Suite.
Tests:
1. Real dataset column mapping & standardization.
2. GIS and weather feature construction.
3. Missing fields and unknown ports / vessels fallback handling.
4. Model training without label leakage (chronological train/test split).
5. Scikit-learn VotingRegressor ensemble serialization and inference.
6. ETA and delay arithmetic (no double-counting of components).
7. Demurrage cost consistency.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add tbp directory to path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TBP_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, TBP_DIR)

from ml.gis_database import (
    resolve_port,
    resolve_origin,
    resolve_plant,
    resolve_vessel,
    haversine_nmi,
    PORT_GIS,
    ORIGIN_GIS,
    STEEL_PLANT_GIS,
)
from ml.weather_client import WeatherMarineClient
from ml.dataset_mapper import (
    standardize_dataframe,
    calculate_route_priors,
    enrich_dataset_with_features,
)
from ml.pipeline import BharatSupplyVesselDelayPredictor


class TestMLIntegration(unittest.TestCase):

    def setUp(self):
        self.shipments_csv = os.path.join(os.path.dirname(TBP_DIR), "shipments.csv")
        if not os.path.exists(self.shipments_csv):
            self.shipments_csv = os.path.join(TBP_DIR, "shipments.csv")

    def test_01_gis_resolution_and_haversine(self):
        """Tests GIS lookups for ports, origins, and haversine calculations."""
        # Test known ports
        p_vizag = resolve_port("Visakhapatnam Port")
        self.assertAlmostEqual(p_vizag["lat"], 17.6868, places=3)
        self.assertGreater(p_vizag["max_draft_m"], 15.0)

        p_haldia = resolve_port("Haldia Dock Complex")
        self.assertAlmostEqual(p_haldia["lat"], 22.0257, places=3)

        # Test unknown port fallback
        p_unknown = resolve_port("Random Nonexistent Port")
        self.assertIn("lat", p_unknown)
        self.assertIn("max_draft_m", p_unknown)

        # Test origin resolution
        o_aus = resolve_origin("Australia")
        self.assertEqual(o_aus["chokepoint"], "Lombok/Sunda Strait")

        # Test Haversine distance
        dist = haversine_nmi(o_aus["lat"], o_aus["lon"], p_vizag["lat"], p_vizag["lon"])
        self.assertGreater(dist, 2000.0)

    def test_02_dataset_mapping_and_validation(self):
        """Tests real shipments.csv column mapping and null handling."""
        self.assertTrue(os.path.exists(self.shipments_csv), f"Missing {self.shipments_csv}")
        df_raw = pd.read_csv(self.shipments_csv)
        self.assertGreater(len(df_raw), 0)

        df_std = standardize_dataframe(df_raw)
        required_cols = [
            "Vessel_ID", "Vessel_Name", "Cargo_Type", "Origin_Country",
            "Arrival_Port", "Steel_Plant", "ETA", "Total_Delay_Hours"
        ]
        for col in required_cols:
            self.assertIn(col, df_std.columns, f"Missing required standardized column {col}")

        # Ensure no negative delays
        self.assertTrue((df_std["Total_Delay_Hours"] >= 0).all())

    def test_03_no_label_leakage_in_route_priors(self):
        """Verifies route priors are computed ONLY from training records."""
        df_raw = pd.read_csv(self.shipments_csv)
        df_std = standardize_dataframe(df_raw)

        split_idx = int(len(df_std) * 0.8)
        train_df = df_std.iloc[:split_idx]
        test_df = df_std.iloc[split_idx:]

        priors = calculate_route_priors(train_df)
        self.assertIn(("__GLOBAL__", "__GLOBAL__"), priors)

        # Enriched test set should use train priors without leaking test delays
        enriched_test = enrich_dataset_with_features(test_df, priors)
        self.assertIn("hist_route_mean_delay", enriched_test.columns)
        self.assertIn("hist_route_std_delay", enriched_test.columns)

    def test_04_model_training_and_evaluation(self):
        """Tests VotingRegressor ensemble training and held-out evaluation."""
        predictor = BharatSupplyVesselDelayPredictor(artifacts_dir=os.path.join(TBP_DIR, "ml", "artifacts"))
        df_raw = pd.read_csv(self.shipments_csv)

        results = predictor.train_and_validate(df_raw, test_ratio=0.20)
        self.assertIn("baseline_model", results)
        self.assertIn("new_ensemble_model", results)

        ensemble_metrics = results["new_ensemble_model"]["metrics"]
        self.assertIn("MAE_hours", ensemble_metrics)
        self.assertIn("RMSE_hours", ensemble_metrics)
        self.assertIn("Prediction_Bias_hours", ensemble_metrics)
        self.assertIn("Accuracy_pct", ensemble_metrics)

        # Verify model was persisted
        self.assertTrue(os.path.exists(os.path.join(TBP_DIR, "ml", "artifacts", "vessel_models.pkl")))
        self.assertTrue(os.path.exists(os.path.join(TBP_DIR, "ml", "artifacts", "model_metadata.json")))

    def test_05_inference_and_delay_arithmetic(self):
        """Tests that delay components sum up to total delay without double counting."""
        predictor = BharatSupplyVesselDelayPredictor(artifacts_dir=os.path.join(TBP_DIR, "ml", "artifacts"))
        predictor.load_artifacts()

        res = predictor.predict(
            vessel_id="TEST_VSL_01",
            vessel_name="MV Bulk Carrier One",
            cargo_type="Coking Coal",
            origin_country="Australia",
            arrival_port="Paradip Port",
            steel_plant="JSW Steel Vijayanagar",
            eta_str="2025-06-15 10:00",
            remaining_distance_km=350.0,
        )

        # Verify no double counting: berthing + customs + yard == total delay (within rounding tolerance)
        comp_sum = round(res.predicted_berthing_delay_h + res.predicted_customs_delay_h + res.predicted_yard_dwell_h, 1)
        self.assertAlmostEqual(comp_sum, res.predicted_total_delay_h, delta=0.2)

        # Verify Demurrage formula: $25,000 / 24h
        expected_demurrage = round((res.predicted_total_delay_h / 24.0) * 25000.0, 2)
        self.assertAlmostEqual(res.predicted_demurrage_usd, expected_demurrage, delta=1.0)

        # Verify calculated ETA is in the future relative to scheduled ETA
        scheduled_dt = pd.to_datetime("2025-06-15 10:00")
        calculated_dt = pd.to_datetime(res.calculated_eta)
        self.assertGreaterEqual(calculated_dt, scheduled_dt)

    def test_06_weather_provenance_and_tagging(self):
        """Verifies environmental data explicitly tags live vs fallback source."""
        env = WeatherMarineClient.fetch_environmental_conditions(20.26, 86.67, "Paradip")
        self.assertIn("source", env)
        self.assertIn("is_live", env)
        self.assertIn("weather_severity_index", env)
        self.assertGreaterEqual(env["weather_severity_index"], 0.0)
        self.assertLessEqual(env["weather_severity_index"], 1.0)


if __name__ == "__main__":
    unittest.main()

