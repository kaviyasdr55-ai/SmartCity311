import requests
import csv
import os
import time

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

OUTPUT_FILE = "data/raw/nyc_311_sample.csv"

TOTAL_RECORDS = 100000
BATCH_SIZE = 50000

FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "incident_zip",
    "borough",
    "latitude",
    "longitude"
]

os.makedirs("data/raw", exist_ok=True)

params_base = {
    "$select": ",".join(FIELDS),
    "$where": "created_date >= '2024-01-01T00:00:00'",
    "$order": "created_date DESC",
    "$limit": BATCH_SIZE
}

total_downloaded = 0

print("Connecting to NYC Open Data API...")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()

    while total_downloaded < TOTAL_RECORDS:

        params = params_base.copy()
        params["$offset"] = total_downloaded

        print(f"Downloading batch: {total_downloaded + 1}")

        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=120
            )

            response.raise_for_status()
            records = response.json()

            if not records:
                print("No more records available.")
                break

            writer.writerows(records)

            total_downloaded += len(records)

            print(
                f"Downloaded: {total_downloaded:,} records"
            )

            if len(records) < BATCH_SIZE:
                break

            time.sleep(1)

        except requests.exceptions.RequestException as error:
            print("Download error:", error)
            break

print("\nDATA COLLECTION COMPLETED")
print("Total records:", total_downloaded)
print("Saved file:", OUTPUT_FILE)

if total_downloaded == 0:
    print("WARNING: No records downloaded. Check API connection.")
