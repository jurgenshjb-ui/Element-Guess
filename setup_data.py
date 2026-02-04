import requests

RAW_URL = "https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json"
OUT_FILE = "PeriodicTableJSON.json"

def main():
    r = requests.get(RAW_URL, timeout=30)
    r.raise_for_status()
    with open(OUT_FILE, "wb") as f:
        f.write(r.content)
    print(f"Saved {OUT_FILE} ({len(r.content):,} bytes)")

if __name__ == "__main__":
    main()
