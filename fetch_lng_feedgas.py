# fetch_lng_feedgas.py
import requests
import sys
from datetime import datetime

"""
LNG Feedgas ist NICHT zuverlässig als CSV verfügbar.
Diese Version ist bewusst DEFENSIV:
- Bei Fehlern: 0.0
- Niemals Text, niemals Exception nach außen
"""

def main():
    try:
        # Platzhalter – EIA bietet hier KEINE stabile öffentliche CSV
        # Sobald eine valide Series-ID existiert, wird sie hier ersetzt
        lng_value = 0.0
        lng_date = "keine neue Meldung"

        # Ausgabe NUR ZAHL (wichtig für GitHub Actions)
        print(f"{lng_value}")

        # Debug optional auf stderr
        print(
            f"[INFO] LNG Feedgas: {lng_value} ({lng_date})",
            file=sys.stderr
        )

    except Exception as e:
        # Absoluter Fallback – garantiert stabil
        print("0.0")
        print(f"[ERROR] LNG Feedgas Fehler: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
