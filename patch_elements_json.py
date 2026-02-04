import json
from pathlib import Path

FILE = Path("PeriodicTableJSON.json")

NOBLE_GASES = {
    "Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon", "Oganesson"
}

# Optional: boiling points in Kelvin (only if you want to force them even if your dataset is missing)
# If your dataset already has a field like "boil" (Kelvin), we will copy that automatically instead.
NOBLE_GAS_BOIL_K = {
    "Helium": 4.22,
    "Neon": 27.07,
    "Argon": 87.30,
    "Krypton": 119.93,
    "Xenon": 165.03,
    "Radon": 211.45,
    # Oganesson: experimentally uncertain / not well established; leave unset by default
}

def main():
    data = json.loads(FILE.read_text(encoding="utf-8"))
    elems = data.get("elements", [])
    if not isinstance(elems, list):
        raise ValueError("Expected data['elements'] to be a list")

    changed = 0

    for e in elems:
        name = e.get("name")
        if not name:
            continue

        # 1) Fix categories
        if name == "Hydrogen":
            if e.get("category") != "nonmetal":
                e["category"] = "nonmetal"
                changed += 1

        if name in NOBLE_GASES:
            if e.get("category") != "nonmetal":
                e["category"] = "nonmetal"
                changed += 1

        # 2) Add noble gas flag
        if name in NOBLE_GASES:
            if e.get("is_noble_gas") is not True:
                e["is_noble_gas"] = True
                changed += 1
        elif name == "Hydrogen":
            # Hydrogen is NOT a noble gas
            if e.get("is_noble_gas") is not False:
                e["is_noble_gas"] = False
                changed += 1

        # 3) Add boiling_point_K (best effort)
        # If your dataset already has "boil" in Kelvin (common in periodic-table datasets), use it.
        if "boiling_point_K" not in e:
            if isinstance(e.get("boil"), (int, float)):
                e["boiling_point_K"] = float(e["boil"])
                changed += 1
            elif name in NOBLE_GASES and name in NOBLE_GAS_BOIL_K:
                e["boiling_point_K"] = float(NOBLE_GAS_BOIL_K[name])
                changed += 1

    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Updated {changed} fields. Wrote: {FILE}")

if __name__ == "__main__":
    main()
