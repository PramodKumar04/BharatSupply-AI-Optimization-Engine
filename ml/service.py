"""
BharatSupply Python ML Prediction Service.
Runs as a lightweight HTTP microservice (built on http.server, zero extra deps)
and supports CLI process invocation for seamless Express/Node.js integration.
"""

import sys
import os
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List

# Ensure package path is accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .pipeline import BharatSupplyVesselDelayPredictor

predictor_instance: BharatSupplyVesselDelayPredictor = None


def get_predictor() -> BharatSupplyVesselDelayPredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = BharatSupplyVesselDelayPredictor()
        # Try loading persisted model, else train on shipments.csv
        if not predictor_instance.load_artifacts():
            shipments_csv = os.path.join(os.path.dirname(PROJECT_ROOT), "shipments.csv")
            if not os.path.exists(shipments_csv):
                shipments_csv = os.path.join(PROJECT_ROOT, "shipments.csv")
            if os.path.exists(shipments_csv):
                import pandas as pd
                print(f"[ML Service] Training initial model on {shipments_csv}...")
                df = pd.read_csv(shipments_csv)
                predictor_instance.train_and_validate(df)
            else:
                print("[ML Service] Warning: shipments.csv not found for initial training.")
    return predictor_instance


class MLRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        predictor = get_predictor()
        path = self.path.split("?")[0]

        if path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "trained": predictor.trained,
                "model_type": "VotingRegressor (RF+GBDT+ExtraTrees+MLP)",
                "evaluation": predictor.evaluation_results.get("new_ensemble_model", {}).get("metrics", {}),
            })
        elif path == "/metrics":
            self._send_json(200, predictor.evaluation_results)
        elif path == "/constraints":
            self._send_json(200, predictor.get_optimizer_constraints())
        else:
            self._send_json(404, {"error": f"Endpoint {path} not found"})

    def do_POST(self):
        predictor = get_predictor()
        path = self.path.split("?")[0]

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as e:
            return self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})

        if path == "/predict":
            try:
                res = predictor.predict(
                    vessel_id=str(body.get("vessel_id") or body.get("id") or "VSL_LIVE"),
                    vessel_name=str(body.get("vessel_name") or body.get("name") or "MV Ocean Steel"),
                    cargo_type=str(body.get("cargo_type") or body.get("material") or "Coking Coal"),
                    origin_country=str(body.get("origin_country") or body.get("origin") or "Australia"),
                    arrival_port=str(body.get("arrival_port") or body.get("destinationPortName") or body.get("destinationPort") or "Visakhapatnam Port"),
                    steel_plant=str(body.get("steel_plant") or body.get("targetPlantName") or body.get("targetPlant") or "Vizag Steel Plant"),
                    eta_str=body.get("eta") or body.get("scheduledETA"),
                    remaining_distance_km=float(body.get("remaining_distance_km", 480.0)),
                    enable_external_news=bool(body.get("enable_external_news", False)),
                )
                self._send_json(200, res.to_dict())
            except Exception as e:
                self._send_json(500, {"error": f"Prediction error: {str(e)}"})

        elif path == "/predict-batch":
            vessels = body.get("vessels", [])
            results = []
            for v in vessels:
                try:
                    res = predictor.predict(
                        vessel_id=str(v.get("vessel_id") or v.get("id") or "VSL_BATCH"),
                        vessel_name=str(v.get("vessel_name") or v.get("name") or "MV Ocean Steel"),
                        cargo_type=str(v.get("cargo_type") or v.get("material") or "Coking Coal"),
                        origin_country=str(v.get("origin_country") or v.get("origin") or "Australia"),
                        arrival_port=str(v.get("arrival_port") or v.get("destinationPortName") or v.get("destinationPort") or "Visakhapatnam Port"),
                        steel_plant=str(v.get("steel_plant") or v.get("targetPlantName") or v.get("targetPlant") or "Vizag Steel Plant"),
                        eta_str=v.get("eta") or v.get("scheduledETA"),
                        remaining_distance_km=float(v.get("remaining_distance_km", 480.0)),
                    )
                    results.append(res.to_dict())
                except Exception as e:
                    results.append({"error": str(e), "vessel_id": v.get("id")})
            self._send_json(200, {"count": len(results), "predictions": results})

        elif path == "/train":
            try:
                import pandas as pd
                records = body.get("data")
                if records and isinstance(records, list):
                    df = pd.DataFrame(records)
                else:
                    shipments_csv = os.path.join(os.path.dirname(PROJECT_ROOT), "shipments.csv")
                    if not os.path.exists(shipments_csv):
                        shipments_csv = os.path.join(PROJECT_ROOT, "shipments.csv")
                    df = pd.read_csv(shipments_csv)
                eval_metrics = predictor.train_and_validate(df)
                self._send_json(200, {"message": "Model trained successfully", "results": eval_metrics})
            except Exception as e:
                self._send_json(500, {"error": f"Training failed: {str(e)}"})
        else:
            self._send_json(404, {"error": f"Endpoint {path} not found"})


def run_cli():
    parser = argparse.ArgumentParser(description="BharatSupply ML Engine CLI")
    parser.add_argument("--server", action="store_true", help="Start HTTP server")
    parser.add_argument("--port", type=int, default=5001, help="Server port (default 5001)")
    parser.add_argument("--predict", type=str, help="JSON string of single vessel payload")
    parser.add_argument("--predict-batch", type=str, help="JSON string of array of vessels")
    parser.add_argument("--train", action="store_true", help="Train model on shipments.csv")
    parser.add_argument("--train-file", type=str, help="Path to CSV file to train on")
    parser.add_argument("--metrics", action="store_true", help="Print model evaluation metrics")
    parser.add_argument("--constraints", action="store_true", help="Print optimizer constraints")
    parser.add_argument("--health", action="store_true", help="Print health status")

    args = parser.parse_args()
    predictor = get_predictor()

    if args.server:
        server_address = ("127.0.0.1", args.port)
        httpd = HTTPServer(server_address, MLRequestHandler)
        print(f"[ML Service] BharatSupply ML Prediction Server running on http://127.0.0.1:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[ML Service] Stopping server...")
            httpd.server_close()
        return

    if args.predict:
        try:
            payload = json.loads(args.predict)
            res = predictor.predict(
                vessel_id=str(payload.get("vessel_id") or payload.get("id") or "VSL_CLI"),
                vessel_name=str(payload.get("vessel_name") or payload.get("name") or "MV Ocean Steel"),
                cargo_type=str(payload.get("cargo_type") or payload.get("material") or "Coking Coal"),
                origin_country=str(payload.get("origin_country") or payload.get("origin") or "Australia"),
                arrival_port=str(payload.get("arrival_port") or payload.get("destinationPortName") or payload.get("destinationPort") or "Visakhapatnam Port"),
                steel_plant=str(payload.get("steel_plant") or payload.get("targetPlantName") or payload.get("targetPlant") or "Vizag Steel Plant"),
                eta_str=payload.get("eta") or payload.get("scheduledETA"),
                remaining_distance_km=float(payload.get("remaining_distance_km", 480.0)),
            )
            print(json.dumps(res.to_dict()))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        return

    if args.predict_batch:
        try:
            payload = json.loads(args.predict_batch)
            vessels = payload if isinstance(payload, list) else payload.get("vessels", [])
            preds = [
                predictor.predict(
                    vessel_id=str(v.get("vessel_id") or v.get("id") or "VSL_CLI"),
                    vessel_name=str(v.get("vessel_name") or v.get("name") or "MV Ocean Steel"),
                    cargo_type=str(v.get("cargo_type") or v.get("material") or "Coking Coal"),
                    origin_country=str(v.get("origin_country") or v.get("origin") or "Australia"),
                    arrival_port=str(v.get("arrival_port") or v.get("destinationPortName") or v.get("destinationPort") or "Visakhapatnam Port"),
                    steel_plant=str(v.get("steel_plant") or v.get("targetPlantName") or v.get("targetPlant") or "Vizag Steel Plant"),
                    eta_str=v.get("eta") or v.get("scheduledETA"),
                ).to_dict()
                for v in vessels
            ]
            print(json.dumps({"count": len(preds), "predictions": preds}))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        return

    if args.train or args.train_file:
        import pandas as pd
        csv_path = args.train_file or os.path.join(os.path.dirname(PROJECT_ROOT), "shipments.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(PROJECT_ROOT, "shipments.csv")
        df = pd.read_csv(csv_path)
        eval_metrics = predictor.train_and_validate(df)
        print(json.dumps({"status": "success", "results": eval_metrics}))
        return

    if args.metrics:
        print(json.dumps(predictor.evaluation_results))
        return

    if args.constraints:
        print(json.dumps(predictor.get_optimizer_constraints()))
        return

    if args.health:
        print(json.dumps({
            "status": "healthy",
            "trained": predictor.trained,
            "model_type": "VotingRegressor Ensemble",
        }))
        return

    parser.print_help()


if __name__ == "__main__":
    run_cli()

