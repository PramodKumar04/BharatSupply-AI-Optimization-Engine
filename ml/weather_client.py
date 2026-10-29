"""
Weather and Marine Environmental Data Client.
Safely fetches real-time atmospheric and sea-state metrics from Open-Meteo,
or falls back gracefully while explicitly tagging data sources as live vs fallback.
"""

import json
import urllib.request
from typing import Dict, Any

class WeatherMarineClient:
    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def fetch_environmental_conditions(cls, lat: float, lon: float, port_name: str = "") -> Dict[str, Any]:
        cache_key = f"{lat:.2f},{lon:.2f}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&current=precipitation,surface_pressure,wind_speed_10m,wind_direction_10m,visibility"
        )
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&current=wave_height,wave_direction,swell_wave_height,ocean_current_velocity"
        )

        env: Dict[str, Any] = {
            "wind_speed_knots": 14.0,
            "wind_dir_deg": 180.0,
            "precipitation_mm": 0.5,
            "pressure_hpa": 1011.0,
            "visibility_km": 10.0,
            "wave_height_m": 1.4,
            "swell_height_m": 1.0,
            "current_knots": 0.8,
            "weather_severity_index": 0.25,
            "source": "simulated_fallback",
            "is_live": False,
            "port_name": port_name,
        }

        live_weather_success = False
        live_marine_success = False

        try:
            req_w = urllib.request.Request(weather_url, headers={"User-Agent": "BharatSupplyETAPredictor/2.0"})
            with urllib.request.urlopen(req_w, timeout=2.0) as resp:
                w_data = json.loads(resp.read().decode("utf-8")).get("current", {})
                if w_data:
                    env["wind_speed_knots"] = round(float(w_data.get("wind_speed_10m", 25.0)) * 0.539957, 2)
                    env["wind_dir_deg"] = float(w_data.get("wind_direction_10m", 180.0))
                    env["precipitation_mm"] = float(w_data.get("precipitation", 0.0))
                    env["pressure_hpa"] = float(w_data.get("surface_pressure", 1012.0))
                    env["visibility_km"] = round(float(w_data.get("visibility", 10000.0)) / 1000.0, 2)
                    live_weather_success = True
        except Exception:
            pass

        try:
            req_m = urllib.request.Request(marine_url, headers={"User-Agent": "BharatSupplyETAPredictor/2.0"})
            with urllib.request.urlopen(req_m, timeout=2.0) as resp:
                m_data = json.loads(resp.read().decode("utf-8")).get("current", {})
                if m_data:
                    if m_data.get("wave_height") is not None:
                        env["wave_height_m"] = round(float(m_data["wave_height"]), 2)
                    if m_data.get("swell_wave_height") is not None:
                        env["swell_height_m"] = round(float(m_data["swell_wave_height"]), 2)
                    if m_data.get("ocean_current_velocity") is not None:
                        env["current_knots"] = round(float(m_data["ocean_current_velocity"]) * 0.539957, 2)
                    live_marine_success = True
        except Exception:
            pass

        if live_weather_success and live_marine_success:
            env["source"] = "live_open_meteo_and_marine_api"
            env["is_live"] = True
        elif live_weather_success:
            env["source"] = "live_open_meteo_weather_api"
            env["is_live"] = True
        else:
            env["source"] = "static_regional_fallback"
            env["is_live"] = False

        wind_norm = min(env["wind_speed_knots"] / 40.0, 1.0)
        wave_norm = min(env["wave_height_m"] / 5.0, 1.0)
        rain_norm = min(env["precipitation_mm"] / 20.0, 1.0)
        vis_penalty = max(0.0, (10.0 - min(env["visibility_km"], 10.0)) / 10.0)
        env["weather_severity_index"] = round(
            0.35 * wind_norm + 0.35 * wave_norm + 0.20 * rain_norm + 0.10 * vis_penalty, 4
        )
        cls._cache[cache_key] = env
        return env

