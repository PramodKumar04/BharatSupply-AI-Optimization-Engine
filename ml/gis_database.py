"""
BharatSupply GIS & Maritime Route Database.
Provides static geospatial coordinates, draft constraints, traffic densities,
and haversine maritime distance calculations without requiring external GIS services.
"""

import math
from typing import Dict, Any

# Ports GIS Database (extended with BharatSupply ports: Haldia, Dhamra, Visakhapatnam, Paradip, JNPT, Chennai)
PORT_GIS: Dict[str, Dict[str, Any]] = {
    "Visakhapatnam Port": {
        "id": "vizag",
        "lat": 17.6868, "lon": 83.2185,
        "max_draft_m": 18.1, "canal_restricted": 0,
        "base_traffic_density": 0.72, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.65,
    },
    "Chennai Port": {
        "id": "chennai",
        "lat": 13.0827, "lon": 80.2707,
        "max_draft_m": 16.5, "canal_restricted": 0,
        "base_traffic_density": 0.85, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.75,
    },
    "JNPT": {
        "id": "jnpt",
        "lat": 18.9499, "lon": 72.9512,
        "max_draft_m": 15.5, "canal_restricted": 1,
        "base_traffic_density": 0.92, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.80,
    },
    "Paradip Port": {
        "id": "paradip",
        "lat": 20.2649, "lon": 86.6750,
        "max_draft_m": 17.1, "canal_restricted": 0,
        "base_traffic_density": 0.78, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.60,
    },
    "Haldia Dock Complex": {
        "id": "haldia",
        "lat": 22.0257, "lon": 88.0629,
        "max_draft_m": 12.5, "canal_restricted": 1,
        "base_traffic_density": 0.82, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.70,
    },
    "Dhamra Port": {
        "id": "dhamra",
        "lat": 20.7833, "lon": 86.9500,
        "max_draft_m": 18.0, "canal_restricted": 0,
        "base_traffic_density": 0.60, "pilotage_compulsory": 1,
        "customs_scrutiny_index": 0.55,
    },
}

# Alias map for port matching
PORT_ALIASES = {
    "paradip": "Paradip Port",
    "paradip port": "Paradip Port",
    "vizag": "Visakhapatnam Port",
    "visakhapatnam": "Visakhapatnam Port",
    "visakhapatnam port": "Visakhapatnam Port",
    "haldia": "Haldia Dock Complex",
    "haldia dock": "Haldia Dock Complex",
    "haldia dock complex": "Haldia Dock Complex",
    "dhamra": "Dhamra Port",
    "dhamra port": "Dhamra Port",
    "jnpt": "JNPT",
    "jawaharlal nehru port": "JNPT",
    "chennai": "Chennai Port",
    "chennai port": "Chennai Port",
}

# Origins GIS Database
ORIGIN_GIS: Dict[str, Dict[str, Any]] = {
    "Australia": {
        "lat": -20.3122, "lon": 118.5753,
        "port_name": "Port Hedland",
        "piracy_risk": 0.15, "war_risk": 0.05,
        "chokepoint": "Lombok/Sunda Strait",
        "chokepoint_lat": -8.75, "chokepoint_lon": 115.70,
        "chokepoint_restriction": 0.45,
    },
    "Indonesia": {
        "lat": -1.2654, "lon": 116.8312,
        "port_name": "Balikpapan Terminal",
        "piracy_risk": 0.45, "war_risk": 0.05,
        "chokepoint": "Strait of Malacca",
        "chokepoint_lat": 2.50, "chokepoint_lon": 101.80,
        "chokepoint_restriction": 0.85,
    },
    "South Africa": {
        "lat": -28.8000, "lon": 32.0833,
        "port_name": "Richards Bay",
        "piracy_risk": 0.35, "war_risk": 0.15,
        "chokepoint": "Mozambique Channel",
        "chokepoint_lat": -16.00, "chokepoint_lon": 41.00,
        "chokepoint_restriction": 0.40,
    },
    "Brazil": {
        "lat": -2.5683, "lon": -44.3708,
        "port_name": "Ponta da Madeira",
        "piracy_risk": 0.30, "war_risk": 0.10,
        "chokepoint": "Cape of Good Hope",
        "chokepoint_lat": -34.35, "chokepoint_lon": 18.47,
        "chokepoint_restriction": 0.55,
    },
    "China": {
        "lat": 39.9300, "lon": 119.6000,
        "port_name": "Qinhuangdao",
        "piracy_risk": 0.20, "war_risk": 0.10,
        "chokepoint": "Strait of Malacca",
        "chokepoint_lat": 2.50, "chokepoint_lon": 101.80,
        "chokepoint_restriction": 0.70,
    },
    "Colombia": {
        "lat": 12.2100, "lon": -71.9700,
        "port_name": "Puerto Bolivar",
        "piracy_risk": 0.25, "war_risk": 0.08,
        "chokepoint": "Cape of Good Hope",
        "chokepoint_lat": -34.35, "chokepoint_lon": 18.47,
        "chokepoint_restriction": 0.50,
    },
    "Russia": {
        "lat": 68.9700, "lon": 33.0800,
        "port_name": "Murmansk",
        "piracy_risk": 0.10, "war_risk": 0.35,
        "chokepoint": "Suez / Bab-el-Mandeb",
        "chokepoint_lat": 12.58, "chokepoint_lon": 43.33,
        "chokepoint_restriction": 0.75,
    },
    "Mozambique": {
        "lat": -25.9692, "lon": 32.5732,
        "port_name": "Maputo",
        "piracy_risk": 0.30, "war_risk": 0.12,
        "chokepoint": "Mozambique Channel",
        "chokepoint_lat": -16.00, "chokepoint_lon": 41.00,
        "chokepoint_restriction": 0.35,
    },
}

ORIGIN_ALIASES = {
    "australia": "Australia",
    "indonesia": "Indonesia",
    "south africa": "South Africa",
    "brazil": "Brazil",
    "china": "China",
    "colombia": "Colombia",
    "russia": "Russia",
    "mozambique": "Mozambique",
    "newcastle": "Australia",
    "hay point": "Australia",
    "port hedland": "Australia",
    "balikpapan": "Indonesia",
    "richards bay": "South Africa",
    "maputo": "Mozambique",
    "ponta da madeira": "Brazil",
    "qinhuangdao": "China",
    "puerto bolivar": "Colombia",
    "murmansk": "Russia",
}

# Steel Plants GIS Database (Includes BharatSupply SAIL plants: Bhilai, Rourkela, Bokaro + Dataset plants)
STEEL_PLANT_GIS: Dict[str, Dict[str, Any]] = {
    "JSW Steel Vijayanagar": {
        "id": "jsw_vijayanagar",
        "lat": 15.1889, "lon": 76.6531,
        "rail_congestion_idx": 0.68,
    },
    "Vizag Steel Plant": {
        "id": "vizag_steel",
        "lat": 17.6316, "lon": 83.1647,
        "rail_congestion_idx": 0.30,
    },
    "Tata Steel Jamshedpur": {
        "id": "tata_jamshedpur",
        "lat": 22.7925, "lon": 86.1842,
        "rail_congestion_idx": 0.74,
    },
    "Bhilai Steel Plant": {
        "id": "bhilai",
        "lat": 21.2094, "lon": 81.3784,
        "rail_congestion_idx": 0.65,
    },
    "Rourkela Steel Plant": {
        "id": "rourkela",
        "lat": 22.2604, "lon": 84.8536,
        "rail_congestion_idx": 0.55,
    },
    "Bokaro Steel Plant": {
        "id": "bokaro",
        "lat": 23.6693, "lon": 86.1511,
        "rail_congestion_idx": 0.60,
    },
}

PLANT_ALIASES = {
    "jsw steel vijayanagar": "JSW Steel Vijayanagar",
    "jsw vijayanagar": "JSW Steel Vijayanagar",
    "vizag steel plant": "Vizag Steel Plant",
    "vizag steel": "Vizag Steel Plant",
    "rinl": "Vizag Steel Plant",
    "tata steel jamshedpur": "Tata Steel Jamshedpur",
    "tata jamshedpur": "Tata Steel Jamshedpur",
    "bhilai": "Bhilai Steel Plant",
    "bhilai steel plant": "Bhilai Steel Plant",
    "rourkela": "Rourkela Steel Plant",
    "rourkela steel plant": "Rourkela Steel Plant",
    "bokaro": "Bokaro Steel Plant",
    "bokaro steel plant": "Bokaro Steel Plant",
}

# Static vessel specifications
VESSEL_STATIC_SPECS: Dict[str, Dict[str, Any]] = {
    "MV Ocean Steel": {
        "vessel_type": "Bulk Carrier", "length_m": 229, "beam_m": 32.2,
        "draft_m": 14.5, "dwt_tons": 82000, "age_years": 6,
        "management_score": 0.88, "engine_power_kw": 12000, "design_speed_knots": 14.5,
    },
    "MV Bulk Carrier One": {
        "vessel_type": "Capesize Bulk", "length_m": 292, "beam_m": 45.0,
        "draft_m": 18.2, "dwt_tons": 180000, "age_years": 12,
        "management_score": 0.76, "engine_power_kw": 18600, "design_speed_knots": 13.8,
    },
    "MV Eastern Glory": {
        "vessel_type": "Supramax Bulk", "length_m": 199, "beam_m": 32.2,
        "draft_m": 13.3, "dwt_tons": 58000, "age_years": 9,
        "management_score": 0.82, "engine_power_kw": 9480, "design_speed_knots": 14.2,
    },
    "MV Coal Express": {
        "vessel_type": "Panamax Bulk", "length_m": 225, "beam_m": 32.2,
        "draft_m": 14.0, "dwt_tons": 76000, "age_years": 15,
        "management_score": 0.70, "engine_power_kw": 10500, "design_speed_knots": 13.5,
    },
    "MV Pacific Miner": {
        "vessel_type": "VLOC", "length_m": 300, "beam_m": 50.0,
        "draft_m": 18.5, "dwt_tons": 208000, "age_years": 5,
        "management_score": 0.91, "engine_power_kw": 20100, "design_speed_knots": 14.8,
    },
    "MV Iron Titan": {
        "vessel_type": "Capesize Bulk", "length_m": 290, "beam_m": 45.0,
        "draft_m": 17.8, "dwt_tons": 175000, "age_years": 11,
        "management_score": 0.79, "engine_power_kw": 17900, "design_speed_knots": 14.0,
    },
}

# Generic fallback vessel profile for unknown vessel names
DEFAULT_VESSEL_SPEC: Dict[str, Any] = {
    "vessel_type": "Panamax Bulk",
    "length_m": 225,
    "beam_m": 32.2,
    "draft_m": 14.2,
    "dwt_tons": 78000,
    "age_years": 10,
    "management_score": 0.80,
    "engine_power_kw": 11500,
    "design_speed_knots": 14.0,
}


def resolve_port(port_name: str) -> Dict[str, Any]:
    if not port_name:
        return PORT_GIS["Visakhapatnam Port"]
    key = PORT_ALIASES.get(port_name.strip().lower(), port_name)
    if key in PORT_GIS:
        return PORT_GIS[key]
    for p_name, data in PORT_GIS.items():
        if port_name.lower() in p_name.lower():
            return data
    return PORT_GIS["Visakhapatnam Port"]


def resolve_origin(origin_name: str) -> Dict[str, Any]:
    if not origin_name:
        return ORIGIN_GIS["Indonesia"]
    key = ORIGIN_ALIASES.get(origin_name.strip().lower(), origin_name)
    if key in ORIGIN_GIS:
        return ORIGIN_GIS[key]
    for o_name, data in ORIGIN_GIS.items():
        if origin_name.lower() in o_name.lower():
            return data
    return ORIGIN_GIS["Indonesia"]


def resolve_plant(plant_name: str) -> Dict[str, Any]:
    if not plant_name:
        return STEEL_PLANT_GIS["Vizag Steel Plant"]
    key = PLANT_ALIASES.get(plant_name.strip().lower(), plant_name)
    if key in STEEL_PLANT_GIS:
        return STEEL_PLANT_GIS[key]
    for p_name, data in STEEL_PLANT_GIS.items():
        if plant_name.lower() in p_name.lower():
            return data
    return STEEL_PLANT_GIS["Vizag Steel Plant"]


def resolve_vessel(vessel_name: str, vessel_age: int = 10) -> Dict[str, Any]:
    if vessel_name in VESSEL_STATIC_SPECS:
        spec = dict(VESSEL_STATIC_SPECS[vessel_name])
        if vessel_age > 0:
            spec["age_years"] = vessel_age
        return spec
    spec = dict(DEFAULT_VESSEL_SPEC)
    if vessel_age > 0:
        spec["age_years"] = vessel_age
    return spec


def haversine_nmi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    dist_km = 2 * r_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return (dist_km * 0.539957) * 1.18

