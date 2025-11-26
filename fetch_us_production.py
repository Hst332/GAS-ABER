# fetch_us_production.py
import random

def fetch_us_production():
    # Hier kann später echte EIA-API implementiert werden
    return round(random.uniform(90, 95), 1)  # in Bcf/day

if __name__ == "__main__":
    print(fetch_us_production())
