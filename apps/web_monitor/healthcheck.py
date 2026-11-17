import json
import os
import sys
import urllib.request

url = os.getenv("HEALTH_URL", "http://127.0.0.1:8088/health")

try:
    with urllib.request.urlopen(url, timeout=2) as response:
        payload = json.load(response)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload.get("status") == "ok" else 1)
except Exception as exc:
    print(f"Servicio no disponible: {exc}", file=sys.stderr)
    raise SystemExit(1)
