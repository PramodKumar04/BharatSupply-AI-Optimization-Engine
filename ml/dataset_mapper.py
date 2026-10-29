"""
Dataset Mapping, Validation, and Feature Enrichment without Label Leakage.
Maps real BharatSupply shipments data to the machine learning feature schema.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .gis_database import (
    resolve_port,
    resolve_origin,
    resolve_plant,
    resolve_vessel,
    haversine_nmi,
)

COLUMN_MAP = {
    # Standard schema mapping
    "vessel_id": "Vessel_ID",
    "vesselid": "Vessel_ID",
    "id": "Vessel_ID",
    "vessel_name": "Vessel_Name",
    "vesselname": "Vessel_Name",
    "name": "Vessel_Name",
    "cargo_type": "Cargo_Type",
    "cargotype": "Cargo_Type",
    "material": "Cargo_Type",
    "origin_country": "Origin_Country",
    "origincountry": "Origin_Country",
    "origin": "Origin_Country",
    "arrival_port": "Arrival_Port",
    "arrivalport": "Arrival_Port",
    "port": "Arrival_Port",
    "destination_port": "Arrival_Port",
    "destinationport": "Arrival_Port",
    "destinationportname": "Arrival_Port",
    "steel_plant": "Steel_Plant",
    "steelplant": "Steel_Plant",
    "plant": "Steel_Plant",
    "targetplant": "Steel_Plant",
    "targetplantname": "Steel_Plant",
    "eta": "ETA",
    "scheduled_eta": "ETA",
    "scheduledeta": "ETA",
    "actual_arrival": "Actual_Arrival",
    "actualarrival": "Actual_Arrival",
    "actual_eta": "Actual_Arrival",
    "actualeta": "Actual_Arrival",
    "total_delay_hours": "Total_Delay_Hours",
    "totaldelayhours": "Total_Delay_Hours",
    "total_delay": "Total_Delay_Hours",
    "delayhours": "Total_Delay_Hours",
    "delay_hours": "Total_Delay_Hours",
    "berthing_delay_hours": "Berthing_Delay_Hours",
    "berthingdelayhours": "Berthing_Delay_Hours",
    "berth_delay": "Berthing_Delay_Hours",
    "customs_clearance_hours": "Customs_Clearance_Hours",
    "customsclearancehours": "Customs_Clearance_Hours",
    "customs_delay": "Customs_Clearance_Hours",
    "yard_dwell_time_hours": "Yard_Dwell_Time_Hours",
    "yarddwelltimehours": "Yard_Dwell_Time_Hours",
    "yard_delay": "Yard_Dwell_Time_Hours",
    "delay_reason": "Delay_Reason",
    "delayreason": "Delay_Reason",
    "reason": "Delay_Reason",
    "demurrage_cost_usd": "Demurrage_Cost_USD",
    "demurragecostusd": "Demurrage_Cost_USD",
    "demurrage": "Demurrage_Cost_USD",
    "shipment_status": "Shipment_Status",
    "status": "Shipment_Status",
}


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names of an uploaded or loaded dataframe."""
    df = df.copy()
    rename_dict = {}
    for col in df.columns:
        norm_col = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if norm_col in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[norm_col]
        elif col in COLUMN_MAP.values():
            rename_dict[col] = col

    df = df.rename(columns=rename_dict)

    # Ensure required categorical columns have fallbacks
    if "Vessel_ID" not in df.columns:
        df["Vessel_ID"] = [f"VSL_{i+1:04d}" for i in range(len(df))]
    if "Vessel_Name" not in df.columns:
        df["Vessel_Name"] = "MV Ocean Steel"
    if "Cargo_Type" not in df.columns:
        df["Cargo_Type"] = "Coking Coal"
    if "Origin_Country" not in df.columns:
        df["Origin_Country"] = "Australia"
    if "Arrival_Port" not in df.columns:
        df["Arrival_Port"] = "Visakhapatnam Port"
    if "Steel_Plant" not in df.columns:
        df["Steel_Plant"] = "Vizag Steel Plant"
    if "ETA" not in df.columns:
        df["ETA"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # If Total_Delay_Hours is missing but delay components exist, sum them
    if "Total_Delay_Hours" not in df.columns:
        comp_sum = 0
        has_comp = False
        for c in ["Berthing_Delay_Hours", "Customs_Clearance_Hours", "Yard_Dwell_Time_Hours"]:
            if c in df.columns:
                comp_sum += pd.to_numeric(df[c], errors="coerce").fillna(0)
                has_comp = True
        if has_comp:
            df["Total_Delay_Hours"] = comp_sum

    if "Delay_Reason" not in df.columns:
        df["Delay_Reason"] = "Port Congestion"

    if "Demurrage_Cost_USD" not in df.columns and "Total_Delay_Hours" in df.columns:
        # Standard maritime demurrage rate: $25,000 / 24h = ~$1,041.67 per delayed hour
        df["Demurrage_Cost_USD"] = (pd.to_numeric(df["Total_Delay_Hours"], errors="coerce").clip(lower=0) / 24.0) * 25000.0

    return df


def calculate_route_priors(train_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Computes route priors STRICTLY on the training set to prevent label leakage.
    Returns a dictionary keyed by (Origin_Country, Arrival_Port).
    """
    priors: Dict[Tuple[str, str], Dict[str, float]] = {}
    grouped = train_df.groupby(["Origin_Country", "Arrival_Port"])["Total_Delay_Hours"]
    global_mean = float(train_df["Total_Delay_Hours"].mean()) if "Total_Delay_Hours" in train_df.columns else 50.0
    global_std = float(train_df["Total_Delay_Hours"].std()) if "Total_Delay_Hours" in train_df.columns and len(train_df) > 1 else 12.0
    if np.isnan(global_std) or global_std <= 0:
        global_std = 12.0

    for (origin, port), group in grouped:
        m = float(group.mean())
        s = float(group.std()) if len(group) > 1 else global_std
        if np.isnan(s) or s <= 0:
            s = global_std
        priors[(str(origin), str(port))] = {"mean": m, "std": s}

    priors[("__GLOBAL__", "__GLOBAL__")] = {"mean": global_mean, "std": global_std}
    return priors


def enrich_dataset_with_features(
    df: pd.DataFrame,
    route_priors: Dict[Tuple[str, str], Dict[str, float]],
    weather_override: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Enriches each record with all 6 PDF factor categories:
    1. Static specs (Vessel, DWT, draft, age, engine)
    2. Dynamic data (Speed, SOG, speed ratio)
    3. Route GIS data (Maritime sea distance, inland plant distance, draft-to-depth ratio)
    4. Environmental data (Wind, wave, rain, weather severity)
    5. Human & management factors (Crew competence, management score)
    6. External unexpected factors (Chokepoints, piracy risk, war risk, customs scrutiny)
    """
    df = df.copy()
    try:
        df["ETA_dt"] = pd.to_datetime(df["ETA"])
    except Exception:
        df["ETA_dt"] = pd.Timestamp.now()

    global_prior = route_priors.get(("__GLOBAL__", "__GLOBAL__"), {"mean": 50.0, "std": 12.0})

    records = []
    for idx, row in df.iterrows():
        v_name = str(row.get("Vessel_Name", "MV Ocean Steel"))
        p_name = str(row.get("Arrival_Port", "Visakhapatnam Port"))
        o_name = str(row.get("Origin_Country", "Australia"))
        s_name = str(row.get("Steel_Plant", "Vizag Steel Plant"))

        p_gis = resolve_port(p_name)
        o_gis = resolve_origin(o_name)
        s_gis = resolve_plant(s_name)
        v_spec = resolve_vessel(v_name)

        # Lookup route priors without leakage
        prior = route_priors.get((o_name, p_name), global_prior)
        hist_mean = prior["mean"]
        hist_std = prior["std"]

        # Geospatial maritime & inland rail calculations
        sea_dist_nmi = haversine_nmi(o_gis["lat"], o_gis["lon"], p_gis["lat"], p_gis["lon"])
        inland_dist_km = haversine_nmi(p_gis["lat"], p_gis["lon"], s_gis["lat"], s_gis["lon"]) * 1.57
        draft_to_depth_ratio = v_spec["draft_m"] / max(10.0, p_gis["max_draft_m"])

        eta_dt = row["ETA_dt"] if isinstance(row["ETA_dt"], pd.Timestamp) else pd.Timestamp.now()
        month = eta_dt.month
        hour = eta_dt.hour
        is_rainy_window = 1 if month in [6, 7, 8, 9, 10, 11] else 0

        # Weather features: use live override if supplied, else seasonal physical baseline
        if weather_override:
            wind_knots = weather_override.get("wind_speed_knots", 14.0)
            wave_m = weather_override.get("wave_height_m", 1.4)
        else:
            wind_knots = 22.0 if is_rainy_window else 12.0
            wave_m = 2.4 if is_rainy_window else 1.2

        sog_knots = max(7.5, v_spec["design_speed_knots"] - 0.12 * max(0, wind_knots - 10) - 0.4 * wave_m)
        speed_strategy_factor = sog_knots / max(10.0, v_spec["design_speed_knots"])
        detention_and_mech_risk = (v_spec["age_years"] / 20.0) * (1.0 - v_spec["management_score"])

        records.append({
            "vessel_type": v_spec["vessel_type"],
            "dwt_tons": v_spec["dwt_tons"],
            "draft_m": v_spec["draft_m"],
            "vessel_age": v_spec["age_years"],
            "management_score": v_spec["management_score"],
            "engine_power_kw": v_spec["engine_power_kw"],
            "hist_route_mean_delay": round(hist_mean, 2),
            "hist_route_std_delay": round(hist_std, 2),
            "ais_sog_knots": round(sog_knots, 2),
            "ais_speed_ratio": round(speed_strategy_factor, 3),
            "sea_dist_nmi": round(sea_dist_nmi, 1),
            "inland_dist_km": round(inland_dist_km, 1),
            "port_traffic_density": p_gis["base_traffic_density"],
            "draft_to_depth_ratio": round(draft_to_depth_ratio, 3),
            "chokepoint_restriction": o_gis["chokepoint_restriction"],
            "rail_congestion_idx": s_gis["rail_congestion_idx"],
            "wind_speed_knots": round(wind_knots, 2),
            "wave_height_m": round(wave_m, 2),
            "crew_competence_idx": v_spec["management_score"],
            "piracy_risk": o_gis["piracy_risk"],
            "war_risk": o_gis["war_risk"],
            "detention_mech_risk": round(detention_and_mech_risk, 4),
            "customs_scrutiny_index": p_gis["customs_scrutiny_index"],
            "eta_month": month,
            "eta_hour": hour,
        })

    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)

