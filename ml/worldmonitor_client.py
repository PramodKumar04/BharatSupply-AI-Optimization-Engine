"""
World Maritime Disruption & Unexpected Factors Client.
Fetches external geopolitical, piracy, port strike, and natural disaster signals.
Provides explicit data provenance (live vs cached fallback) and avoids double-counting delay.
"""

import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Any

from .gis_database import resolve_origin, resolve_port

class WorldMonitorNewsClient:
    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def fetch_unexpected_factors(
        cls,
        origin_country: str,
        arrival_port: str,
        chokepoint: str,
        enable_network: bool = True
    ) -> Dict[str, Any]:
        cache_key = f"{origin_country}|{arrival_port}|{chokepoint}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        o_gis = resolve_origin(origin_country)
        p_gis = resolve_port(arrival_port)

        result: Dict[str, Any] = {
            "war_risk_score": o_gis.get("war_risk", 0.05),
            "piracy_risk_score": o_gis.get("piracy_risk", 0.15),
            "detention_and_strike_risk": 0.18,
            "accidents_and_canal_risk": 0.15,
            "customs_inspection_surge": p_gis.get("customs_scrutiny_index", 0.65),
            "seismic_tsunami_risk": 0.05,
            "unexpected_delay_penalty_hours": 0.0,
            "unexpected_severity_index": 0.22,
            "live_headlines": [],
            "source": "static_gis_priors",
            "is_live": False,
        }

        if not enable_network:
            cls._cache[cache_key] = result
            return result

        headlines: List[Dict[str, str]] = []
        port_short = arrival_port.split()[0] if arrival_port else "Port"

        # 1. Query Google News RSS for maritime disruptions
        query_terms = f'maritime OR shipping OR "{port_short}" OR "{chokepoint}" OR "port congestion" OR piracy'
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_terms)}&hl=en-US&gl=US&ceid=US:en"
        try:
            req = urllib.request.Request(rss_url, headers={"User-Agent": "BharatSupplyMaritime/2.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                xml_bytes = resp.read()
                root = ET.fromstring(xml_bytes)
                for item in root.findall(".//item")[:5]:
                    title = item.findtext("title", default="")
                    pub_date = item.findtext("pubDate", default="")
                    link = item.findtext("link", default="")
                    if title:
                        headlines.append({
                            "title": title,
                            "published": pub_date[:22] if pub_date else "Live",
                            "source": "Google News RSS",
                            "url": link,
                        })
        except Exception:
            pass

        # 2. Query USGS Seismic API for major earthquakes in shipping lanes
        usgs_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
        try:
            req_u = urllib.request.Request(usgs_url, headers={"User-Agent": "BharatSupplyMaritime/2.0"})
            with urllib.request.urlopen(req_u, timeout=2.0) as resp:
                u_data = json.loads(resp.read().decode("utf-8"))
                quakes_near_lanes = 0
                for feat in u_data.get("features", [])[:20]:
                    coords = feat.get("geometry", {}).get("coordinates", [0, 0])
                    mag = feat.get("properties", {}).get("mag", 0) or 0
                    place = feat.get("properties", {}).get("place", "")
                    if 20 <= coords[0] <= 130 and -35 <= coords[1] <= 30 and mag >= 5.0:
                        quakes_near_lanes += 1
                        if len(headlines) < 6:
                            headlines.append({
                                "title": f"USGS Seismic Alert: M{mag:.1f} near {place}",
                                "published": "Live 24h",
                                "source": "USGS Seismic",
                                "url": "https://earthquake.usgs.gov",
                            })
                if quakes_near_lanes > 0:
                    result["seismic_tsunami_risk"] = min(0.65, 0.10 + 0.12 * quakes_near_lanes)
        except Exception:
            pass

        if headlines:
            result["source"] = "live_external_feed"
            result["is_live"] = True
            combined_text = " ".join(h["title"].lower() for h in headlines)
            if any(w in combined_text for w in ["war", "missile", "military", "conflict", "red sea", "houthi"]):
                result["war_risk_score"] = min(0.90, result["war_risk_score"] + 0.25)
            if any(w in combined_text for w in ["piracy", "pirate", "hijack", "armed", "malacca"]):
                result["piracy_risk_score"] = min(0.90, result["piracy_risk_score"] + 0.22)
            if any(w in combined_text for w in ["strike", "detention", "union", "protest", "lockout"]):
                result["detention_and_strike_risk"] = min(0.85, result["detention_and_strike_risk"] + 0.30)
            if any(w in combined_text for w in ["collision", "grounding", "accident", "fire", "blocked"]):
                result["accidents_and_canal_risk"] = min(0.85, result["accidents_and_canal_risk"] + 0.28)
            if any(w in combined_text for w in ["customs", "tariff", "inspection", "scrutiny", "delay"]):
                result["customs_inspection_surge"] = min(0.95, result["customs_inspection_surge"] + 0.15)
        else:
            result["source"] = "cached_regional_intelligence"
            result["is_live"] = False
            headlines = [
                {
                    "title": f"Chokepoint Traffic Status: Normal navigational queuing at {chokepoint}",
                    "published": "Historical Benchmark",
                    "source": "Static Maritime Baseline",
                    "url": "#",
                }
            ]

        severity = (
            0.25 * result["war_risk_score"]
            + 0.20 * result["piracy_risk_score"]
            + 0.20 * result["detention_and_strike_risk"]
            + 0.15 * result["accidents_and_canal_risk"]
            + 0.15 * result["customs_inspection_surge"]
            + 0.05 * result["seismic_tsunami_risk"]
        )
        result["unexpected_severity_index"] = round(severity, 3)
        result["live_headlines"] = headlines
        cls._cache[cache_key] = result
        return result

