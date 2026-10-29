"""
Machine Learning Pipeline for Multi-Source Vessel Delay Prediction.
Builds the scikit-learn ColumnTransformer, VotingRegressor ensemble (RF, GBDT, ExtraTrees, MLP),
and reason classifier. Persists models, preprocessors, and metadata.
"""

import os
import json
import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .gis_database import (
    PORT_GIS,
    ORIGIN_GIS,
    STEEL_PLANT_GIS,
    VESSEL_STATIC_SPECS,
    haversine_nmi,
    resolve_port,
    resolve_origin,
    resolve_plant,
    resolve_vessel,
)
from .weather_client import WeatherMarineClient
from .worldmonitor_client import WorldMonitorNewsClient
from .rl_pathfinder import MaritimeRLPathFinder
from .dataset_mapper import (
    standardize_dataframe,
    calculate_route_priors,
    enrich_dataset_with_features,
)

CAT_COLS = ["Cargo_Type", "Origin_Country", "Arrival_Port", "Steel_Plant", "vessel_type"]
NUM_COLS = [
    "dwt_tons", "draft_m", "vessel_age", "management_score", "engine_power_kw",
    "hist_route_mean_delay", "hist_route_std_delay", "ais_sog_knots", "ais_speed_ratio",
    "sea_dist_nmi", "inland_dist_km", "port_traffic_density", "draft_to_depth_ratio",
    "chokepoint_restriction", "rail_congestion_idx", "wind_speed_knots", "wave_height_m",
    "crew_competence_idx", "piracy_risk", "war_risk", "detention_mech_risk",
    "customs_scrutiny_index", "eta_month", "eta_hour",
]


@dataclass
class SinglePrediction:
    vessel_id: str
    vessel_name: str
    origin_country: str
    arrival_port: str
    steel_plant: str
    selected_paradigm: str
    selected_framework: str
    selection_rationale: str
    ml_test_mae_hours: float
    predicted_berthing_delay_h: float
    predicted_customs_delay_h: float
    predicted_yard_dwell_h: float
    predicted_total_delay_h: float
    predicted_remaining_sea_hours: float
    calculated_eta: str
    predicted_demurrage_usd: float
    predicted_primary_delay_reason: str
    rl_optimal_speed_knots: float
    rl_route_detour_ratio: float
    rl_grid_path: List[Tuple[int, int]]
    weather_summary: Dict[str, Any]
    worldmonitor_summary: Dict[str, Any]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id,
            "vessel_name": self.vessel_name,
            "origin_country": self.origin_country,
            "arrival_port": self.arrival_port,
            "steel_plant": self.steel_plant,
            "selected_paradigm": self.selected_paradigm,
            "selected_framework": self.selected_framework,
            "selection_rationale": self.selection_rationale,
            "ml_test_mae_hours": self.ml_test_mae_hours,
            "predicted_berthing_delay_h": self.predicted_berthing_delay_h,
            "predicted_customs_delay_h": self.predicted_customs_delay_h,
            "predicted_yard_dwell_h": self.predicted_yard_dwell_h,
            "predicted_total_delay_h": self.predicted_total_delay_h,
            "predicted_remaining_sea_hours": self.predicted_remaining_sea_hours,
            "calculated_eta": self.calculated_eta,
            "predicted_demurrage_usd": self.predicted_demurrage_usd,
            "predicted_primary_delay_reason": self.predicted_primary_delay_reason,
            "rl_optimal_speed_knots": self.rl_optimal_speed_knots,
            "rl_route_detour_ratio": self.rl_route_detour_ratio,
            "rl_grid_path": self.rl_grid_path,
            "weather_summary": self.weather_summary,
            "worldmonitor_summary": self.worldmonitor_summary,
            "confidence_score": self.confidence_score,
            # Backward-compatibility fields for existing frontend consumers
            "predictedDelay": self.predicted_total_delay_h,
            "confidence": self.confidence_score,
            "factors": [
                {"name": f"Berthing Queue ({self.predicted_berthing_delay_h}h)", "impact": "high" if self.predicted_berthing_delay_h > 15 else "medium", "direction": "increase"},
                {"name": f"Customs Scrutiny ({self.predicted_customs_delay_h}h)", "impact": "high" if self.predicted_customs_delay_h > 12 else "medium", "direction": "increase"},
                {"name": f"Yard Handling ({self.predicted_yard_dwell_h}h)", "impact": "medium", "direction": "increase"},
                {"name": f"Primary Cause: {self.predicted_primary_delay_reason}", "impact": "high", "direction": "increase"},
            ],
        }


class BharatSupplyVesselDelayPredictor:
    """
    Core Machine Learning engine replacing client-side mock trees.
    Ensemble: scikit-learn VotingRegressor (RF + GBDT + ExtraTrees + MLP)
    Plus RL Q-learning route hazard search and dynamic model selection.
    """

    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.rl_engine = MaritimeRLPathFinder()
        self.cat_cols = CAT_COLS
        self.num_cols = NUM_COLS
        self.route_priors: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.models: Dict[str, Pipeline] = {}
        self.reason_classifier: Optional[Pipeline] = None
        self.evaluation_results: Dict[str, Any] = {}
        self.trained = False

    def _build_preprocessor(self) -> ColumnTransformer:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, self.cat_cols),
                ("num", num_pipe, self.num_cols),
            ]
        )

    def _build_ensemble_regressor(self) -> Pipeline:
        ensemble = VotingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)),
                ("gbdt", GradientBoostingRegressor(n_estimators=80, learning_rate=0.06, max_depth=4, random_state=42)),
                ("ert", ExtraTreesRegressor(n_estimators=80, max_depth=8, random_state=42)),
                ("mlp", MLPRegressor(hidden_layer_sizes=(48, 24), alpha=0.1, max_iter=300, random_state=42)),
            ],
            weights=[0.30, 0.30, 0.25, 0.15],
        )
        return Pipeline([("prep", self._build_preprocessor()), ("reg", ensemble)])

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        bias = float(np.mean(y_pred - y_true))
        denom = np.maximum(np.abs(y_true), 15.0)
        error_rate_pct = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
        acc_pct = max(0.0, 100.0 - error_rate_pct)
        within_12h = float(np.mean(np.abs(y_true - y_pred) <= 12.0) * 100.0)
        within_24h = float(np.mean(np.abs(y_true - y_pred) <= 24.0) * 100.0)

        # R² score calculation
        tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
        rss = float(np.sum((y_true - y_pred) ** 2))
        r2 = max(0.0, 1.0 - (rss / tss)) if tss > 0.01 else 0.0

        return {
            "MAE_hours": round(mae, 2),
            "RMSE_hours": round(rmse, 2),
            "Prediction_Bias_hours": round(bias, 2),
            "Error_Rate_pct": round(error_rate_pct, 1),
            "Accuracy_pct": round(acc_pct, 1),
            "Within_12h_pct": round(within_12h, 1),
            "Within_24h_pct": round(within_24h, 1),
            "R2_Score": round(r2, 3),
        }

    def train_and_validate(self, df_raw: pd.DataFrame, test_ratio: float = 0.20) -> Dict[str, Any]:
        """
        Trains models strictly using chronological validation without label leakage.
        Compares Baseline model vs New VotingRegressor model on identical held-out examples.
        """
        df = standardize_dataframe(df_raw)

        # Chronological ordering by ETA
        try:
            df["ETA_parsed"] = pd.to_datetime(df["ETA"])
            df = df.sort_values("ETA_parsed").reset_index(drop=True)
        except Exception:
            pass

        n_total = len(df)
        n_train = max(5, int(n_total * (1.0 - test_ratio)))
        train_df_raw = df.iloc[:n_train].copy()
        test_df_raw = df.iloc[n_train:].copy()

        # Step 1: Compute route priors STRICTLY on the training set
        self.route_priors = calculate_route_priors(train_df_raw)

        # Step 2: Enrich train and test sets using the training route priors
        train_df = enrich_dataset_with_features(train_df_raw, self.route_priors)
        test_df = enrich_dataset_with_features(test_df_raw, self.route_priors)

        X_train = train_df[self.cat_cols + self.num_cols]
        y_train = train_df["Total_Delay_Hours"].astype(float).values

        X_test = test_df[self.cat_cols + self.num_cols]
        y_test = test_df["Total_Delay_Hours"].astype(float).values

        # ── Baseline Model (Dummy Mean Regressor) ─────────────────────
        baseline_pipe = Pipeline([
            ("prep", self._build_preprocessor()),
            ("reg", DummyRegressor(strategy="mean")),
        ])
        baseline_pipe.fit(X_train, y_train)
        baseline_test_preds = baseline_pipe.predict(X_test)
        baseline_metrics = self._compute_metrics(y_test, baseline_test_preds)

        # ── New VotingRegressor Ensemble Model ────────────────────────
        ensemble_pipe = self._build_ensemble_regressor()
        ensemble_pipe.fit(X_train, y_train)
        ensemble_test_preds = ensemble_pipe.predict(X_test)
        ensemble_metrics = self._compute_metrics(y_test, ensemble_test_preds)

        # ── Train sub-target models on training split ─────────────────
        targets = [
            "Total_Delay_Hours",
            "Berthing_Delay_Hours",
            "Customs_Clearance_Hours",
            "Yard_Dwell_Time_Hours",
            "Demurrage_Cost_USD",
        ]
        self.models = {}
        for t in targets:
            if t in train_df.columns and train_df[t].notnull().sum() > 3:
                pipe = self._build_ensemble_regressor()
                pipe.fit(X_train, train_df[t].astype(float).values)
                self.models[t] = pipe
            else:
                self.models[t] = ensemble_pipe

        # Train Delay Reason Classifier
        if "Delay_Reason" in train_df.columns:
            clf_pipe = Pipeline([
                ("prep", self._build_preprocessor()),
                ("clf", RandomForestClassifier(n_estimators=80, random_state=42)),
            ])
            clf_pipe.fit(X_train, train_df["Delay_Reason"].astype(str).values)
            self.reason_classifier = clf_pipe

        self.evaluation_results = {
            "dataset_total_samples": n_total,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "baseline_model": {
                "name": "Historical Mean Baseline (Framework 1 Dummy)",
                "metrics": baseline_metrics,
            },
            "new_ensemble_model": {
                "name": "Multi-Source VotingRegressor Ensemble (RF + GBDT + ExtraTrees + MLP)",
                "metrics": ensemble_metrics,
            },
            "mae_improvement_hours": round(baseline_metrics["MAE_hours"] - ensemble_metrics["MAE_hours"], 2),
            "rmse_improvement_hours": round(baseline_metrics["RMSE_hours"] - ensemble_metrics["RMSE_hours"], 2),
            "trained_at": pd.Timestamp.now().isoformat(),
        }

        self.trained = True
        self.save_artifacts()
        return self.evaluation_results

    def save_artifacts(self) -> None:
        """Persists trained model pipelines and route priors."""
        model_file = os.path.join(self.artifacts_dir, "vessel_models.pkl")
        meta_file = os.path.join(self.artifacts_dir, "model_metadata.json")

        payload = {
            "models": self.models,
            "reason_classifier": self.reason_classifier,
            "route_priors": {f"{k[0]}|{k[1]}": v for k, v in self.route_priors.items()},
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "evaluation_results": self.evaluation_results,
        }
        with open(model_file, "wb") as f:
            pickle.dump(payload, f)

        with open(meta_file, "w") as f:
            json.dump({
                "evaluation_results": self.evaluation_results,
                "cat_cols": self.cat_cols,
                "num_cols": self.num_cols,
                "model_type": "VotingRegressor (RF+GBDT+ExtraTrees+MLP)",
            }, f, indent=2)

    def load_artifacts(self) -> bool:
        """Loads serialized model pipelines from disk."""
        model_file = os.path.join(self.artifacts_dir, "vessel_models.pkl")
        if not os.path.exists(model_file):
            return False
        try:
            with open(model_file, "rb") as f:
                payload = pickle.load(f)
            self.models = payload.get("models", {})
            self.reason_classifier = payload.get("reason_classifier")
            raw_priors = payload.get("route_priors", {})
            self.route_priors = {}
            for k, v in raw_priors.items():
                parts = k.split("|")
                if len(parts) == 2:
                    self.route_priors[(parts[0], parts[1])] = v
            self.evaluation_results = payload.get("evaluation_results", {})
            self.trained = True
            return True
        except Exception as e:
            print(f"[ML Pipeline] Artifact load failed: {e}")
            return False

    def get_optimizer_constraints(self) -> Dict[str, Any]:
        """Provides dynamic constraints for the MILP optimizer."""
        metrics = self.evaluation_results.get("new_ensemble_model", {}).get("metrics", {})
        mean_delay = float(self.route_priors.get(("__GLOBAL__", "__GLOBAL__"), {}).get("mean", 48.0))
        std_delay = float(self.route_priors.get(("__GLOBAL__", "__GLOBAL__"), {}).get("std", 14.0))

        return {
            "monsoonPenalty": 1.35,
            "congestionWeight": 0.32,
            "weatherRiskFactor": 1.25,
            "vesselAgeThreshold": 15,
            "confidenceLevel": min(0.96, metrics.get("Accuracy_pct", 85.0) / 100.0),
            "meanTrainedDelay": round(mean_delay, 1),
            "stdTrainedDelay": round(std_delay, 1),
            "testMaeHours": metrics.get("MAE_hours", 8.5),
            "modelType": "VotingRegressor Ensemble",
            "lastTrained": self.evaluation_results.get("trained_at", pd.Timestamp.now().isoformat()),
        }

    def predict(
        self,
        vessel_id: str,
        vessel_name: str,
        cargo_type: str = "Coking Coal",
        origin_country: str = "Australia",
        arrival_port: str = "Visakhapatnam Port",
        steel_plant: str = "Vizag Steel Plant",
        eta_str: Optional[str] = None,
        remaining_distance_km: float = 480.0,
        enable_external_news: bool = False,
    ) -> SinglePrediction:
        """Executes full inference pipeline for a single shipment."""
        if not self.trained:
            if not self.load_artifacts():
                # Self-train on default shipments.csv if present
                shipments_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shipments.csv")
                if os.path.exists(shipments_path):
                    raw_df = pd.read_csv(shipments_path)
                    self.train_and_validate(raw_df)
                else:
                    raise RuntimeError("Predictor is not trained and shipments.csv was not found.")

        p_gis = resolve_port(arrival_port)
        o_gis = resolve_origin(origin_country)
        v_spec = resolve_vessel(vessel_name)

        # 1. Environmental conditions (tagged live vs fallback)
        live_env = WeatherMarineClient.fetch_environmental_conditions(p_gis["lat"], p_gis["lon"], arrival_port)

        # 2. External unexpected factors (optional live news feed)
        live_news = WorldMonitorNewsClient.fetch_unexpected_factors(
            origin_country=origin_country,
            arrival_port=arrival_port,
            chokepoint=o_gis.get("chokepoint", "Strait"),
            enable_network=enable_external_news,
        )

        # 3. Feature enrichment for single sample
        now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        eta_clean = eta_str if eta_str else now_str

        single_df = pd.DataFrame([{
            "Vessel_ID": vessel_id,
            "Vessel_Name": vessel_name,
            "Cargo_Type": cargo_type,
            "Origin_Country": origin_country,
            "Arrival_Port": arrival_port,
            "Steel_Plant": steel_plant,
            "ETA": eta_clean,
            "Actual_Arrival": eta_clean,
            "Plant_Arrival_Date": eta_clean,
            "Total_Delay_Hours": 50.0,
            "Delay_Reason": "Port Congestion",
        }])

        enriched_df = enrich_dataset_with_features(
            single_df,
            self.route_priors,
            weather_override={
                "wind_speed_knots": live_env["wind_speed_knots"],
                "wave_height_m": live_env["wave_height_m"],
            }
        )
        row = enriched_df.iloc[0]
        X_input = pd.DataFrame([row[self.cat_cols + self.num_cols]])

        # 4. Framework 1: Scikit-Learn Ensemble Regressions
        ml_total = float(self.models["Total_Delay_Hours"].predict(X_input)[0])
        ml_berth = float(self.models["Berthing_Delay_Hours"].predict(X_input)[0]) if "Berthing_Delay_Hours" in self.models else ml_total * 0.45
        ml_customs = float(self.models["Customs_Clearance_Hours"].predict(X_input)[0]) if "Customs_Clearance_Hours" in self.models else ml_total * 0.25
        ml_yard = float(self.models["Yard_Dwell_Time_Hours"].predict(X_input)[0]) if "Yard_Dwell_Time_Hours" in self.models else ml_total * 0.30

        # Delay Reason
        if self.reason_classifier is not None:
            pred_reason = str(self.reason_classifier.predict(X_input)[0])
        else:
            pred_reason = "Port Congestion"

        # 5. Framework 2: RL Route Hazard & MCMC Speed Analysis
        cost_grid = self.rl_engine.build_cost_grid(
            weather_severity=live_env["weather_severity_index"],
            traffic_density=float(row["port_traffic_density"]),
            chokepoint_restriction=float(row["chokepoint_restriction"]),
            piracy_risk=float(row["piracy_risk"]),
            war_risk=float(row["war_risk"]),
        )
        rl_path, mean_hazard = self.rl_engine.train_and_find_path(cost_grid)
        rl_speed = self.rl_engine.metropolis_hastings_speed(
            design_speed_knots=v_spec["design_speed_knots"],
            mean_hazard=mean_hazard,
            weather_severity=live_env["weather_severity_index"],
        )
        ideal_steps = (self.rl_engine.grid_size - 1) * math.sqrt(2)
        actual_steps = sum(
            math.hypot(rl_path[i + 1][0] - rl_path[i][0], rl_path[i + 1][1] - rl_path[i][1])
            for i in range(len(rl_path) - 1)
        )
        rl_detour = max(1.0, actual_steps / max(1.0, ideal_steps))

        # Dynamic Paradigm Selection
        weather_sev = live_env["weather_severity_index"]
        port_traffic = float(row["port_traffic_density"])
        news_risk = live_news["unexpected_severity_index"]

        if remaining_distance_km <= 600.0 and (weather_sev >= 0.30 or news_risk >= 0.35):
            paradigm = "RL_Primary_with_ML_Port_Head"
            framework = "Framework 2 (Adaptive RL Trajectory + ML Port-Head)"
            rationale = f"Approach phase ({remaining_distance_km:.0f} km) with active marine risk ({weather_sev:.2f})."
            final_total_h = max(0.0, ml_total * 0.95 + (rl_detour - 1.0) * 14.0)
        elif weather_sev >= 0.40 or port_traffic >= 0.85:
            paradigm = "Hybrid_Adaptive_RL_ML"
            framework = "Framework 1 + Framework 2 Fusion"
            rationale = f"Elevated dynamic variance (Weather={weather_sev:.2f}, Port Traffic={port_traffic:.2f})."
            final_total_h = max(0.0, ml_total * 0.90 + (rl_detour - 1.0) * 10.0)
        else:
            paradigm = "Machine_Learning_Ensemble"
            framework = "Framework 1 (Multi-Source VotingRegressor Ensemble)"
            rationale = f"Stable sea-state conditions across {remaining_distance_km:.0f} km transit."
            final_total_h = max(0.0, ml_total)

        final_total_h = round(final_total_h, 1)

        # Scale delay components proportionally so: berthing + customs + yard = final_total_h (no double counting!)
        comp_sum = max(0.1, ml_berth + ml_customs + ml_yard)
        scale = final_total_h / comp_sum
        final_berth_h = round(ml_berth * scale, 1)
        final_customs_h = round(ml_customs * scale, 1)
        final_yard_h = round(final_total_h - final_berth_h - final_customs_h, 1)
        if final_yard_h < 0:
            final_yard_h = 0.0

        # Remaining travel time and ETA arithmetic
        remaining_nmi = (remaining_distance_km * 0.539957) * rl_detour
        remaining_sea_hours = round(remaining_nmi / max(6.0, rl_speed), 1)

        try:
            base_eta_dt = pd.to_datetime(eta_clean)
        except Exception:
            base_eta_dt = pd.Timestamp.now()
        calculated_eta_dt = base_eta_dt + pd.Timedelta(hours=float(final_total_h))
        calculated_eta_str = calculated_eta_dt.strftime("%Y-%m-%d %H:%M")

        # Demurrage Cost: $25,000 / day standard cost model
        final_demurrage_usd = round((final_total_h / 24.0) * 25000.0, 2)

        # Confidence metric
        test_acc = self.evaluation_results.get("new_ensemble_model", {}).get("metrics", {}).get("Accuracy_pct", 88.0)
        conf_score = round(max(0.65, min(0.98, (test_acc / 100.0) - (0.05 if weather_sev > 0.4 else 0.0))), 2)

        test_mae = self.evaluation_results.get("new_ensemble_model", {}).get("metrics", {}).get("MAE_hours", 8.5)

        return SinglePrediction(
            vessel_id=vessel_id,
            vessel_name=vessel_name,
            origin_country=origin_country,
            arrival_port=arrival_port,
            steel_plant=steel_plant,
            selected_paradigm=paradigm,
            selected_framework=framework,
            selection_rationale=rationale,
            ml_test_mae_hours=test_mae,
            predicted_berthing_delay_h=final_berth_h,
            predicted_customs_delay_h=final_customs_h,
            predicted_yard_dwell_h=final_yard_h,
            predicted_total_delay_h=final_total_h,
            predicted_remaining_sea_hours=remaining_sea_hours,
            calculated_eta=calculated_eta_str,
            predicted_demurrage_usd=final_demurrage_usd,
            predicted_primary_delay_reason=pred_reason,
            rl_optimal_speed_knots=round(rl_speed, 2),
            rl_route_detour_ratio=round(rl_detour, 3),
            rl_grid_path=rl_path,
            weather_summary=live_env,
            worldmonitor_summary=live_news,
            confidence_score=conf_score,
        )

