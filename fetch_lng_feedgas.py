# fetch_lng_feedgas.py
import os
import requests
import sys

EIA_API_KEY = os.getenv("EIA_API_KEY")

SERIES_ID = "NG.N8110US2.M"  # Monthly LNG exports, Bcf/d (STABIL)

def main():
    if not EIA_API_KEY:
        print("0.0")
        print("[WARN] Kein EIA_API_KEY gesetzt", file=sys.stderr)
        return

    url = (
        "https://api.eia.gov/v2/seriesid/"
        f"{SERIES_ID}/data/"
        f"?api_key={EIA_API_KEY}&length=1"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        value = data["response"]["data"][0]["value"]
        period = data["response"]["data"][0]["period"]

        value = float(value)

        # WICHTIG: stdout = NUR ZAHL
        print(f"{value}")

        # Debug auf stderr
        print(
            f"[INFO] LNG Feedgas {value} Bcf/d (Periode {period}, monatlich)",
            file=sys.stderr
        )

    except Exception as e:
        print("0.0")
        print(f"[ERROR] LNG Feedgas Fallback: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
