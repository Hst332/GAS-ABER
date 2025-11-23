name: Run Gas Price Forecast

on:
  workflow_dispatch:

jobs:
  run-forecast:
    runs-on: ubuntu-24.04
    env:
      EIA_API_KEY: ${{ secrets.EIA_API_KEY }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.10

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch EIA Storage
        id: fetch_eia
        run: |
          echo "Fetching EIA Storage..."
          STORAGE=$(python3 - <<'PYTHON'
import os
import requests

api_key = os.environ.get("EIA_API_KEY")
if not api_key:
    raise Exception("EIA_API_KEY not set in environment")

# v2 API call for weekly US natural gas storage
url = f"https://api.eia.gov/v2/seriesid/NG.WKST.W?api_key={api_key}"
resp = requests.get(url)
if resp.status_code != 200:
    raise Exception(f"EIA API returned {resp.status_code}: {resp.text}")

data = resp.json()
if 'response' not in data or 'data' not in data['response'] or len(data['response']['data']) == 0:
    raise Exception("EIA API returned no data")

value = data['response']['data'][0]['value']
print(value)
PYTHON
          )
          echo "EIA_STORAGE=$STORAGE" >> $GITHUB_ENV
          echo "EIA Storage fetched: $STORAGE"

      - name: Run Gas Price Forecast
        run: |
          echo "Running gas price forecast..."
          python gas_price_forecast.py \
            --eia-storage $EIA_STORAGE \
            --us-production <US_PRODUCTION_VALUE> \
            --lng-feedgas <LNG_FEEDGAS_VALUE> \
            --futures-curve <FUTURES_CURVE_VALUE> \
            --cot-managed-money <COT_MANAGED_MONEY_VALUE> \
            > forecast_output.txt

      - name: Upload forecast output
        uses: actions/upload-artifact@v4
        with:
          name: forecast_output
          path: forecast_output.txt
