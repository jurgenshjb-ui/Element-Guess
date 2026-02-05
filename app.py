from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st
import streamlit.components.v1 as components

from element_tiles import periodic_table

# =========================================================
# Public app metadata
# =========================================================
APP_VERSION = "v1.3.0"
GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO"

# =========================================================
# Production / Debug gating
# Secrets (Streamlit Cloud -> App settings -> Secrets):
# STREAMLIT_ENV = "prod"
# ALLOW_DEBUG = true
# =========================================================
IS_PRODUCTION = (
    os.environ.get("STREAMLIT_ENV") == "prod"
    or st.secrets.get("STREAMLIT_ENV", "dev") == "prod"
)
ALLOW_DEBUG = bool(st.secrets.get("ALLOW_DEBUG", False))
SHOW_DEBUG_UI = (not IS_PRODUCTION) or ALLOW_DEBUG


# =========================================================
# Data model
# =========================================================
@dataclass(frozen=True)
class Element:
    name: str
    symbol: str
    atomic_number: int
    category3: str
    group: Optional[int]
    period: Optional[int]
    state: str  # solid/liquid/gas/unknown from JSON "phase"
    summary: str
    discovered_by: str
    named_by: str
    source: str
    is_noble_gas: bool
    boiling_point_K: Optional[float]
    melting_point_K: Optional[float]
    bohr_model_image: str


# =========================================================
# Chemistry corrections (in-code)
# =========================================================
NOBLE_GAS_NAMES = {"Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon", "Oganesson"}
CATEGORY_OVERRIDES = {"Hydrogen": "nonmetal", **{n: "nonmetal" for n in NOBLE_GAS_NAMES}}


def normalize_state(phase: str) -> str:
    t = (phase or "").lower().strip()
    return t if t in ("solid", "liquid", "gas") else "unknown"


def normalize_category3(raw_category: str, name: str) -> str:
    if name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[name]

    t = (raw_category or "").lower()
    if "metalloid" in t:
        return "metalloid"
    if "nonmetal" in t:
        return "nonmetal"
    return "metal"


def compute_is_noble_gas(name: str, group: Optional[int]) -> bool:
    if name in NOBLE_GAS_NAMES:
        return True
    return group == 18


# =========================================================
# Radioactivity rule (game rule)
# Tc (43), Pm (61), and all Z >= 84
# =========================================================
def is_radioactive(e: Element) -> bool:
    return (e.atomic_number in (43, 61)) or (e.atomic_number >= 84)


def origin_natural_vs_synthetic(e: Element) -> str:
    if e.atomic_number >= 93:
        return "synthetic"
    if e.atomic_number in (43, 61):
        return "synthetic"
    return "natural"


def is_lanthanoid_or_actinoid(e: Element) -> bool:
    return (57 <= e.atomic_number <= 71) or (89 <= e.atomic_number <= 103)


def group_family(e: Element) -> str:
    if is_lanthanoid_or_actinoid(e):
        return "lanthanoid or actinoid"
    g = e.group
    return {
        1: "alkali metal (group 1)",
        2: "alkaline earth metal (group 2)",
        17: "halogen (group 17)",
        18: "noble gas (group 18)",
    }.get(g, "other/none")


def block_of(e: Element) -> str:
    n, g = e.atomic_number, e.group
    if 57 <= n <= 71 or 89 <= n <= 103:
        return "f-block"
    if g in (1, 2) or e.name in ("Hydrogen", "Helium"):
        return "s-block"
    if g and 3 <= g <= 12:
        return "d-block"
    if g and 13 <= g <= 18:
        return "p-block"
    return "unknown"


def metal_group(e: Element) -> str:
    n = e.atomic_number
    g = e.group
    if 57 <= n <= 71 or 89 <= n <= 103:
        return "inner transition metals"
    if g == 1 and e.name != "Hydrogen":
        return "alkali metals"
    if g == 2:
        return "alkaline earth metals"
    if g and 3 <= g <= 12:
        return "transition metals"
    if e.category3 == "metal" and block_of(e) == "p-block":
        return "post-transition metals"
    return "other/unknown"


def atomic_band(n: int) -> str:
    if n <= 10:
        return "1–10"
    if n <= 20:
        return "11–20"
    if n <= 40:
        return "21–40"
    if n <= 60:
        return "41–60"
    if n <= 80:
        return "61–80"
    return "81–118"


# =========================================================
# Load elements
# =========================================================
@st.cache_data
def load_elements(path: str = "PeriodicTableJSON.json") -> List[Element]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw = data["elements"]
    out: List[Element] = []

    for r in raw:
        name = (r.get("name") or "").strip()
        symbol = (r.get("symbol") or "").strip()
        number = int(r.get("number"))
        group = r.get("group")
        period = r.get("period")

        state = normalize_state(r.get("phase"))
        category3 = normalize_category3(r.get("category"), name)

        is_ng = bool(r.get("is_noble_gas")) if ("is_noble_gas" in r) else compute_is_noble_gas(name, group)

        # Boiling point (K)
        bp = r.get("boiling_point_K", None)
        if bp is None and isinstance(r.get("boil"), (int, float)):
            bp = float(r["boil"])
        elif isinstance(bp, (int, float)):
            bp = float(bp)
        else:
            bp = None

        # Melting point (K)
        mp = r.get("melting_point_K", None)
        if mp is None and isinstance(r.get("melt"), (int, float)):
            mp = float(r["melt"])
        elif isinstance(mp, (int, float)):
            mp = float(mp)
        else:
            mp = None

        bohr_img = (r.get("bohr_model_image") or "").strip()

        out.append(
            Element(
                name=name,
                symbol=symbol,
                atomic_number=number,
                category3=category3,
                group=group,
                period=period,
                state=state,
                summary=(r.get("summary") or "").strip(),
                discovered_by=(r.get("discovered_by") or "").strip(),
                named_by=(r.get("named_by") or "").strip(),
                source=(r.get("source") or "").strip(),
                is_noble_gas=is_ng,
                boiling_point_K=bp,
                melting_point_K=mp,
                bohr_model_image=bohr_img,
            )
        )

    return sorted(out, key=lambda e: e.atomic_number)


# =========================================================
# Clues
# =========================================================
CLUES: Dict[str, str] = {
    "category3": "Category (metal / nonmetal / metalloid)",
    "group_family": "Major group family",
    "block": "Electron block (s / p / d / f)",
    "period": "Period",
    "band": "Atomic number range",
    "state": "State at room temp",
    "noble_gas": "Noble gas",
    "boil_split": "Boiling point (K)",
    "melt_split": "Melting point (K)",
    "radioactive": "Radioactive",
    "origin": "Origin (natural / synthetic)",
    "metal_group": "Metal group",
}

CLUE_ORDER = [
    "category3",
    "group_family",
    "block",
    "period",
    "band",
    "state",        # phase first
    "noble_gas",    # only eligible if state == gas
    "boil_split",
    "melt_split",
    "radioactive",
    "origin",
    "metal_group",
]


def prop(e: Element, k: str) -> str:
    if k == "category3":
        return e.category3
    if k == "state":
        return e.state
    if k == "period":
        return str(e.period)
    if k == "group_family":
        return group_family(e)
    if k == "block":
        return block_of(e)
    if k == "band":
        return atomic_band(e.atomic_number)
    if k == "noble_gas":
        return "Yes" if e.is_noble_gas else "No"
    if k == "radioactive":
        return "Yes" if is_radioactive(e) else "No"
    if k == "origin":
        return origin_natural_vs_synthetic(e)
    if k == "metal_group":
        return metal_group(e)
    return "unknown"


def _temp_value_K(e: Element, kind: str) -> Optional[float]:
    if kind == "boil":
        return e.boiling_point_K
    if kind == "melt":
        return e.melting_point_K
    return None


def matches(e: Element, revealed: Dict[str, str]) -> bool:
    for k, v in revealed.items():
        if k in ("boil_split", "melt_split"):
            kind = "boil" if k == "boil_split" else "melt"
            val = _temp_value_K(e, kind)
            if val is None:
                return False
            op, xs = v.split("|", 1)
            x = float(xs)
            if op == "le" and not (val <= x):
                return False
            if op == "gt" and not (val > x):
                return False
            continue

        if prop(e, k) != v:
            return False
    return True


def allowed_clue_keys(revealed: Dict[str, str], attempt: int) -> List[str]:
    keys = [k for k in CLUE_ORDER if k not in revealed]

    # Radioactive only after guess 3
    if attempt < 3 and "radioactive" in keys:
        keys.remove("radioactive")

    # Temperature splits only after guess 3
    if attempt < 3:
        if "boil_split" in keys:
            keys.remove("boil_split")
        if "melt_split" in keys:
            keys.remove("melt_split")

    # Temperature family mutual exclusion: only ever reveal ONE of them
    if "boil_split" in revealed and "melt_split" in keys:
        keys.remove("melt_split")
    if "melt_split" in revealed and "boil_split" in keys:
        keys.remove("boil_split")

    # origin only after radioactive yes
    if "origin" in keys and revealed.get("radioactive") != "Yes":
        keys.remove("origin")

    # metal_group only after metal
    if "metal_group" in keys and revealed.get("category3") != "metal":
        keys.remove("metal_group")

    # noble gas only if state == gas
    if "noble_gas" in keys and revealed.get("state") != "gas":
        keys.remove("noble_gas")

    return keys


def debug_unrevealed_keys(revealed: Dict[str, str]) -> List[str]:
    # Debug ignores gameplay gating, but keeps "only one temperature split" logic to avoid confusion
    keys = [k for k in CLUE_ORDER if k not in revealed]
    if "boil_split" in revealed and "melt_split" in keys:
        keys.remove("melt_split")
    if "melt_split" in revealed and "boil_split" in keys:
        keys.remove("boil_split")
    return keys


# =========================================================
# Entropy scoring
# =========================================================
def _entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _distribution_for_clue(
    current_candidates: List[Element],
    clue_key: str,
    temp_split_thresholds: Dict[str, Tuple[Optional[float], int, int]],
) -> Tuple[Dict[str, int], int]:
    """
    Returns (value_counts, total_considered).
    For categorical clues: counts of each value.
    For temp split clues: counts for buckets ["≤X", ">X", "unknown"] (unknown counted).
    """
    if clue_key not in ("boil_split", "melt_split"):
        counts: Dict[str, int] = {}
        for e in current_candidates:
            v = prop(e, clue_key)
            counts[v] = counts.get(v, 0) + 1
        return counts, len(current_candidates)

    kind = "boil" if clue_key == "boil_split" else "melt"
    x, known, total = temp_split_thresholds.get(kind, (None, 0, len(current_candidates)))
    counts = {"≤X": 0, ">X": 0, "unknown": 0}
    if x is None:
        return counts, total

    for e in current_candidates:
        val = _temp_value_K(e, kind)
        if val is None:
            counts["unknown"] += 1
        elif val <= x:
            counts["≤X"] += 1
        else:
            counts[">X"] += 1

    return counts, total


# =========================================================
# Temperature split helpers (rounded to nearest 100 K)
# + guardrail: require >=70% known among current candidates
# =========================================================
def _choose_temp_split_threshold(current_candidates: List[Element], kind: str) -> Tuple[Optional[float], int, int]:
    vals = [(_temp_value_K(e, kind)) for e in current_candidates]
    known_vals = sorted([v for v in vals if v is not None])
    total = len(current_candidates)
    known = len(known_vals)

    if total == 0 or known < 6:
        return None, known, total

    # If too many unknowns, skip this clue (avoid "gotcha" elimination)
    if (known / total) < 0.70:
        return None, known, total

    median = known_vals[len(known_vals) // 2]
    x = round(median / 100.0) * 100.0
    return float(x), known, total


def _secret_temp_bucket(secret: Element, clue_key: str, x: float) -> Optional[str]:
    kind = "boil" if clue_key == "boil_split" else "melt"
    val = _temp_value_K(secret, kind)
    if val is None:
        return None
    return "≤X" if val <= x else ">X"


def _count_candidates_for_secret_temp_value(
    secret: Element, current_candidates: List[Element], clue_key: str, x: float
) -> Optional[Tuple[str, int]]:
    bucket = _secret_temp_bucket(secret, clue_key, x)
    if bucket is None:
        return None
    kind = "boil" if clue_key == "boil_split" else "melt"
    cnt = 0
    for e in current_candidates:
        val = _temp_value_K(e, kind)
        if val is None:
            continue
        if bucket == "≤X" and val <= x:
            cnt += 1
        elif bucket == ">X" and val > x:
            cnt += 1
    return bucket, cnt


def _deterministic_temp_tiebreak(game_mode: str, seed: Optional[int]) -> int:
    """
    Returns 0 or 1 deterministically per game.
    Daily: based on date. Endless: based on seed.
    """
    if game_mode == "Daily":
        d = dt.date.today().toordinal()
        return d % 2
    # Endless
    s = seed or 0
    return s % 2


def choose_next_clue_entropy(
    secret: Element,
    elements: List[Element],
    revealed: Dict[str, str],
    attempt: int,
    game_mode: str,
    seed: Optional[int],
) -> Optional[Tuple[str, str, int, float]]:
    """
    Choose next clue by:
      1) Only consider clues that STRICTLY NARROW (0 < cnt < current_n), otherwise ignore.
      2) Score by highest entropy of value distribution among current candidates.
      3) Tie-break by:
         a) larger secret-bucket candidate count (keeps next pool wider)
         b) larger minimum bucket size (avoids 99/1)
         c) deterministic temperature tie-break when boil vs melt are still tied
    """
    remaining_keys = allowed_clue_keys(revealed, attempt)
    if not remaining_keys:
        return None

    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    if current_n <= 1:
        return None

    # Precompute possible temp thresholds
    temp_thresholds = {
        "boil": _choose_temp_split_threshold(current_candidates, "boil"),
        "melt": _choose_temp_split_threshold(current_candidates, "melt"),
    }

    best: Optional[Tuple[str, str, int, float, int, int]] = None
    # best tuple: (key, value, cnt_secret, entropy, min_bucket, temp_tiebreak_rank)

    for k in remaining_keys:
        if k in ("boil_split", "melt_split"):
            kind = "boil" if k == "boil_split" else "melt"
            x, known, total = temp_thresholds[kind]
            if x is None:
                continue

            res = _count_candidates_for_secret_temp_value(secret, current_candidates, k, x)
            if res is None:
                continue
            bucket, cnt = res

            # Must strictly narrow
            if not (0 < cnt < current_n):
                continue

            counts_dict, _ = _distribution_for_clue(current_candidates, k, temp_thresholds)
            ent = _entropy_from_counts(list(counts_dict.values()))
            min_bucket = min([c for c in counts_dict.values() if c > 0], default=0)

            op = "le" if bucket == "≤X" else "gt"
            v = f"{op}|{int(x)}"

            # deterministic tie-break only when needed (boil vs melt)
            temp_rank = _deterministic_temp_tiebreak(game_mode, seed)
            # For ranking, let boil be 0, melt be 1; flip by temp_rank to alternate
            # If temp_rank==0 => prefer boil; if 1 => prefer melt
            prefer_melt = (temp_rank == 1)
            this_is_melt = (k == "melt_split")
            tie_rank = 0 if (this_is_melt == prefer_melt) else 1

            cand = (k, v, cnt, ent, min_bucket, tie_rank)
        else:
            secret_value = prop(secret, k)
            cnt = sum(1 for e in current_candidates if prop(e, k) == secret_value)

            # Must strictly narrow
            if not (0 < cnt < current_n):
                continue

            counts_dict, _ = _distribution_for_clue(current_candidates, k, temp_thresholds)
            ent = _entropy_from_counts(list(counts_dict.values()))
            min_bucket = min([c for c in counts_dict.values() if c > 0], default=0)
            cand = (k, secret_value, cnt, ent, min_bucket, 0)

        if best is None:
            best = cand
        else:
            # Compare by entropy first, then keep next pool wide, then avoid tiny buckets, then temp tie rank
            _, _, cnt_b, ent_b, min_b, tie_b = best
            _, _, cnt_c, ent_c, min_c, tie_c = cand

            if ent_c > ent_b + 1e-9:
                best = cand
            elif abs(ent_c - ent_b) <= 1e-9:
                if cnt_c > cnt_b:
                    best = cand
                elif cnt_c == cnt_b:
                    if min_c > min_b:
                        best = cand
                    elif min_c == min_b:
                        # only meaningful for temp split vs temp split
                        if tie_c < tie_b:
                            best = cand

    if best is None:
        # If nothing strictly narrows, return None (we skip useless clues)
        return None

    k, v, cnt, ent, _, _ = best
    return k, v, cnt, ent


def reveal_next_clue(secret: Element, elements: List[Element], attempt: int) -> None:
    nxt = choose_next_clue_entropy(
        secret=secret,
        elements=elements,
        revealed=st.session_state.revealed,
        attempt=attempt,
        game_mode=st.session_state.game_mode,
        seed=st.session_state.seed,
    )
    if not nxt:
        return
    k, v, _, _ = nxt
    if k not in st.session_state.revealed:
        st.session_state.revealed[k] = v
        st.session_state.revealed_order.append(k)


# =========================================================
# Definitions panel
# =========================================================
def definitions_panel():
    with st.expander("📘 Definitions (Categories, Groups, Blocks, Radioactive)", expanded=False):
        st.markdown(
            """
**Broad Categories**
- **Metals**: shiny, malleable, good conductors; tend to lose electrons.
- **Nonmetals**: poorer conductors; tend to gain electrons.
- **Metalloids**: intermediate properties; often semiconductors.

**Major Groups / Families**
- **Alkali metals (Group 1)**: very reactive metals with one valence electron.
- **Alkaline earth metals (Group 2)**: reactive metals with two valence electrons.
- **Halogens (Group 17)**: reactive nonmetals that gain one electron.
- **Noble gases (Group 18)**: stable, mostly unreactive nonmetals.
- **Lanthanoid or actinoid**: elements in **57–71** or **89–103**.

**Electron Blocks**
- **s-block**: Groups 1–2 (plus H, He).
- **p-block**: Groups 13–18.
- **d-block**: Transition metals (Groups 3–12).
- **f-block**: Lanthanoids & actinoids.

**Radioactive (game rule)**
- **Tc (43)** and **Pm (61)** are radioactive
- **All elements with Z ≥ 84** are radioactive

**Temperature split clues**
- Reveals **Boiling point: ≤ X K** or **> X K**, *or* **Melting point: ≤ X K** or **> X K**
- Only one temperature clue appears per game (never both)
- **X** is chosen to split remaining candidates roughly in half (rounded to nearest **100 K**)
"""
        )


# =========================================================
# Tooltips
# =========================================================
def tooltip_basic(e: Element) -> str:
    return f"{e.name} ({e.symbol}) — atomic #{e.atomic_number}"


def tooltip_guess(e: Element, revealed: Dict[str, str]) -> str:
    lines = [tooltip_basic(e), ""]
    if not revealed:
        return "\n".join(lines + ["No clues revealed yet."])
    for k, v in revealed.items():
        if k in ("boil_split", "melt_split"):
            op, xs = v.split("|", 1)
            sign = "≤" if op == "le" else ">"
            ok = matches(e, {k: v})
            lines.append(f"{'✅' if ok else '❌'} {CLUES[k]}: {sign} {xs} K")
        else:
            ok = (prop(e, k) == v)
            lines.append(f"{'✅' if ok else '❌'} {CLUES.get(k, k)}: {v}")
    return "\n".join(lines)


# =========================================================
# Daily / Endless selection
# =========================================================
def pick_daily(elements: List[Element], date: Optional[dt.date] = None) -> Element:
    date = date or dt.date.today()
    start = dt.date(2026, 1, 1)
    idx = abs((date - start).days) % len(elements)
    return elements[idx]


def pick_endless(elements: List[Element], seed: int) -> Element:
    return random.Random(seed).choice(elements)


# =========================================================
# Special clue (Higher / Lower) with toggle
# =========================================================
def compute_special_atomic_clue(
    enabled: bool,
    elements: List[Element],
    revealed: Dict[str, str],
    guessed: Set[int],
    guesses_left: int,
    secret: Element,
    last_valid_guess_atomic: Optional[int],
) -> Optional[str]:
    if not enabled:
        return None
    if last_valid_guess_atomic is None or guesses_left <= 0:
        return None

    remaining = [e for e in elements if e.atomic_number not in guessed and matches(e, revealed)]
    if len(remaining) <= guesses_left:
        return None

    if secret.atomic_number > last_valid_guess_atomic:
        return "Atomic number: **Higher** than your last valid guess"
    if secret.atomic_number < last_valid_guess_atomic:
        return "Atomic number: **Lower** than your last valid guess"
    return "Atomic number: **Equal** to your last valid guess"


# =========================================================
# Debug helpers (distributions + %)
# =========================================================
def _distribution_table(current_candidates: List[Element], clue_key: str) -> List[dict]:
    counts: Dict[str, int] = {}

    if clue_key in ("boil_split", "melt_split"):
        kind = "boil" if clue_key == "boil_split" else "melt"
        # For debug, show rounded-to-100 buckets (not split ≤/>)
        for e in current_candidates:
            val = _temp_value_K(e, kind)
            if val is None:
                v = "unknown"
            else:
                bucket = int(round(val / 100.0) * 100)
                v = f"{bucket} K"
            counts[v] = counts.get(v, 0) + 1
    else:
        for e in current_candidates:
            v = prop(e, clue_key)
            counts[v] = counts.get(v, 0) + 1

    total = max(len(current_candidates), 1)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))

    out = []
    for v, c in ordered:
        pct = (c / total) * 100.0
        out.append({"value": v, "count": c, "percent": f"{pct:.1f}%"})
    return out


def show_debug_panel(secret: Element, elements: List[Element], revealed: Dict[str, str]):
    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    st.write(f"**Current candidates (matching revealed):** {current_n}")

    # 1) Secret-value impact + delta reduction (debug ignores gating)
    rows = []
    # Precompute temp thresholds for "if revealed now" impact
    temp_thresholds = {
        "boil": _choose_temp_split_threshold(current_candidates, "boil"),
        "melt": _choose_temp_split_threshold(current_candidates, "melt"),
    }

    for k in debug_unrevealed_keys(revealed):
        if k in ("boil_split", "melt_split"):
            kind = "boil" if k == "boil_split" else "melt"
            x, known, total = temp_thresholds[kind]
            if x is None:
                continue
            res = _count_candidates_for_secret_temp_value(secret, current_candidates, k, x)
            if res is None:
                continue
            bucket, cnt = res
            secret_value = f"{'≤' if bucket=='≤X' else '>'} {int(x)} K"
        else:
            secret_value = prop(secret, k)
            cnt = sum(1 for e in current_candidates if prop(e, k) == secret_value)

        reduction = current_n - cnt
        reduction_pct = (reduction / max(current_n, 1)) * 100.0
        rows.append(
            {
                "clue_key": k,
                "clue": CLUES.get(k, k),
                "secret_value": secret_value,
                "candidates_if_revealed": cnt,
                "reduction": reduction,
                "reduction_%": f"{reduction_pct:.1f}%",
            }
        )

    rows.sort(key=lambda r: (-r["candidates_if_revealed"], r["clue_key"]))
    if rows:
        st.markdown("**1) Unrevealed clue impact (secret value → candidates) + reduction:**")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("**2) Distribution per unrevealed clue (value → count, % of candidates):**")
    for k in debug_unrevealed_keys(revealed):
        with st.expander(f"{CLUES.get(k,k)} distribution", expanded=False):
            st.dataframe(_distribution_table(current_candidates, k), use_container_width=True, hide_index=True)


# =========================================================
# Tiles
# =========================================================
def build_positions(elements: List[Element]):
    main: Dict[Tuple[int, int], Element] = {}
    lanth: List[Element] = []
    actin: List[Element] = []

    for e in elements:
        if 57 <= e.atomic_number <= 71:
            lanth.append(e)
        elif 89 <= e.atomic_number <= 103:
            actin.append(e)
        elif e.period is not None and e.group is not None:
            main[(e.period, e.group)] = e

    lanth.sort(key=lambda x: x.atomic_number)
    actin.sort(key=lambda x: x.atomic_number)
    return main, lanth, actin


def make_tiles(
    elements: List[Element],
    revealed: Dict[str, str],
    guessed: Set[int],
    secret_atomic: int,
    difficulty: str,
    last_guess_atomic: Optional[int],
    win_anim_pending: bool,
    status: str,
) -> List[dict]:
    main, lanth, actin = build_positions(elements)
    easy = (difficulty == "Easy")
    normal_locks = (difficulty == "Normal")

    current_hint_set: Set[int] = set()
    if easy and revealed:
        for e in elements:
            if e.atomic_number not in guessed and matches(e, revealed):
                current_hint_set.add(e.atomic_number)

    prev_hint_set: Set[int] = set(st.session_state.prev_hint_set)
    new_hint_set = current_hint_set - prev_hint_set
    st.session_state.prev_hint_set = current_hint_set

    def status_of(e: Element) -> str:
        # reveal secret on loss
        if status == "lost" and e.atomic_number == secret_atomic and e.atomic_number not in guessed:
            return "lost"
        if e.atomic_number in guessed:
            if e.atomic_number == secret_atomic:
                return "correct"
            return "close" if matches(e, revealed) else "bad"
        if easy and revealed and matches(e, revealed):
            return "hint"
        return "none"

    def locked_of(e: Element) -> bool:
        return bool(normal_locks and revealed and (e.atomic_number not in guessed) and (not matches(e, revealed)))

    def tooltip_of(e: Element) -> str:
        if status == "lost" and e.atomic_number == secret_atomic:
            return f"✅ The correct element was {e.name} ({e.symbol}) — atomic #{e.atomic_number}"
        return tooltip_guess(e, revealed) if e.atomic_number in guessed else tooltip_basic(e)

    def tile_dict(e: Element, row: str, pos: Optional[int], group: Optional[int], period: Optional[int]) -> dict:
        return dict(
            symbol=e.symbol,
            name=e.name,
            atomic=e.atomic_number,
            group=group,
            period=period,
            row=row,
            pos=pos,
            status=status_of(e),
            locked=locked_of(e),
            tooltip=tooltip_of(e),
            isLastGuess=(last_guess_atomic is not None and e.atomic_number == last_guess_atomic),
            isNewHint=(e.atomic_number in new_hint_set),
            isWin=(win_anim_pending and e.atomic_number == secret_atomic),
        )

    tiles: List[dict] = []
    for (p, g), e in main.items():
        tiles.append(tile_dict(e, row="main", pos=None, group=g, period=p))
    for i, e in enumerate(lanth[:15]):
        tiles.append(tile_dict(e, row="lanth", pos=i, group=None, period=None))
    for i, e in enumerate(actin[:15]):
        tiles.append(tile_dict(e, row="actin", pos=i, group=None, period=None))
    return tiles


# =========================================================
# Share helpers
# =========================================================
def emoji_for_guess(feedback: str) -> str:
    return {"bad": "🟥", "close": "🟧", "correct": "🟩"}.get(feedback, "⬜")


def build_share_text() -> str:
    mode = st.session_state.game_mode
    difficulty = st.session_state.difficulty
    today = dt.date.today().isoformat()
    guesses_used = len(st.session_state.guesses)
    max_guesses = st.session_state.max_guesses
    rows = [emoji_for_guess(x) for x in st.session_state.guess_feedback]
    grid = "\n".join(rows) if rows else ""
    header = f"Element Guess — {mode}"
    if mode == "Daily":
        header += f" ({today})"
    header += f"\nDifficulty: {difficulty}\nScore: {guesses_used}/{max_guesses}"
    return f"{header}\n\n{grid}".strip()


def js_escape_for_template(s: str) -> str:
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def copy_to_clipboard_button(text: str, label: str = "📋 Copy results"):
    escaped = js_escape_for_template(text)
    components.html(
        f"""
        <div style="display:flex; gap:10px; align-items:center; font-family:system-ui;">
          <button id="copyBtn"
                  style="padding:8px 12px; border-radius:10px; border:1px solid rgba(0,0,0,.2);
                         background:white; font-weight:700; cursor:pointer;">
            {label}
          </button>
          <span id="msg" style="color:#374151; font-size:13px;"></span>
        </div>
        <script>
          const text = `{escaped}`;
          const btn = document.getElementById("copyBtn");
          const msg = document.getElementById("msg");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(text);
              msg.textContent = "Copied!";
              setTimeout(() => msg.textContent = "", 1200);
            }} catch (e) {{
              msg.textContent = "Copy failed (browser blocked clipboard).";
            }}
          }});
        </script>
        """,
        height=60,
    )


def copy_link_button():
    components.html(
        """
        <div style="display:flex; gap:10px; align-items:center; font-family:system-ui;">
          <button id="copyLinkBtn"
                  style="padding:8px 12px; border-radius:10px; border:1px solid rgba(0,0,0,.2);
                         background:white; font-weight:700; cursor:pointer;">
            🔗 Copy link
          </button>
          <span id="msg" style="color:#374151; font-size:13px;"></span>
        </div>
        <script>
          const btn = document.getElementById("copyLinkBtn");
          const msg = document.getElementById("msg");
          btn.addEventListener("click", async () => {
            try {
              const url = (window.parent && window.parent.location && window.parent.location.href)
                          ? window.parent.location.href
                          : window.location.href;
              await navigator.clipboard.writeText(url);
              msg.textContent = "Copied!";
              setTimeout(() => msg.textContent = "", 1200);
            } catch (e) {
              msg.textContent = "Copy failed (browser blocked clipboard).";
            }
          });
        </script>
        """,
        height=60,
    )


# =========================================================
# Debug undo snapshots
# =========================================================
def push_snapshot():
    snap = dict(
        guesses=list(st.session_state.guesses),
        guess_feedback=list(st.session_state.guess_feedback),
        revealed=dict(st.session_state.revealed),
        revealed_order=list(st.session_state.revealed_order),
        attempt=int(st.session_state.attempt),
        last_valid_guess_atomic=st.session_state.last_valid_guess_atomic,
        status=st.session_state.status,
        prev_hint_set=set(st.session_state.prev_hint_set),
        last_guess_atomic=st.session_state.last_guess_atomic,
        invalid_atomic=st.session_state.invalid_atomic,
        win_anim_pending=st.session_state.win_anim_pending,
        stats_recorded=st.session_state.stats_recorded,
        board_nonce=st.session_state.board_nonce,
    )
    st.session_state.history.append(snap)


def undo_snapshot():
    if not st.session_state.history:
        return
    snap = st.session_state.history.pop()
    st.session_state.guesses = snap["guesses"]
    st.session_state.guess_feedback = snap["guess_feedback"]
    st.session_state.revealed = snap["revealed"]
    st.session_state.revealed_order = snap["revealed_order"]
    st.session_state.attempt = snap["attempt"]
    st.session_state.last_valid_guess_atomic = snap["last_valid_guess_atomic"]
    st.session_state.status = snap["status"]
    st.session_state.prev_hint_set = snap["prev_hint_set"]
    st.session_state.last_guess_atomic = snap["last_guess_atomic"]
    st.session_state.invalid_atomic = snap["invalid_atomic"]
    st.session_state.win_anim_pending = snap["win_anim_pending"]
    st.session_state.stats_recorded = snap["stats_recorded"]
    st.session_state.board_nonce = snap["board_nonce"]


# =========================================================
# Stats (production-safe, per-session)
# =========================================================
def init_stats():
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "current_streak": 0,
            "best_streak": 0,
            "total_guesses_in_wins": 0,
            "win_guess_counts": [],
        }


def record_win(guesses_used: int):
    s = st.session_state.stats
    s["games_played"] += 1
    s["wins"] += 1
    s["current_streak"] += 1
    s["best_streak"] = max(s["best_streak"], s["current_streak"])
    s["total_guesses_in_wins"] += guesses_used
    s["win_guess_counts"].append(guesses_used)


def record_loss():
    s = st.session_state.stats
    s["games_played"] += 1
    s["losses"] += 1
    s["current_streak"] = 0


def stats_panel():
    with st.expander("📊 Stats", expanded=False):
        s = st.session_state.stats
        if s["games_played"] == 0:
            st.caption("No games played yet.")
            return

        win_rate = (s["wins"] / s["games_played"]) * 100
        st.markdown(
            f"""
**Games played:** {s["games_played"]}  
**Wins:** {s["wins"]}  
**Losses:** {s["losses"]}  
**Win rate:** {win_rate:.1f}%  

**Current streak:** {s["current_streak"]}  
**Best streak:** {s["best_streak"]}  
"""
        )

        if s["wins"] > 0:
            avg_guesses = s["total_guesses_in_wins"] / s["wins"]
            st.markdown(f"**Average guesses (wins):** {avg_guesses:.2f}")

            dist: Dict[int, int] = {}
            for g in s["win_guess_counts"]:
                dist[g] = dist.get(g, 0) + 1

            st.markdown("**Guess distribution (wins):**")
            for g in sorted(dist):
                st.markdown(f"- {g} guesses: {dist[g]}")


# =========================================================
# State helpers
# =========================================================
def _bump_board_nonce():
    st.session_state.board_nonce = int(st.session_state.get("board_nonce", 0)) + 1


def start_new_game(elements: List[Element], game_mode: str):
    # Fully reset game state (including UI-only markers)
    st.session_state.status = "playing"
    st.session_state.guesses = []
    st.session_state.guess_feedback = []
    st.session_state.revealed = {}
    st.session_state.revealed_order = []
    st.session_state.attempt = 0
    st.session_state.max_guesses = 7
    st.session_state.last_click_nonce = None
    st.session_state.last_valid_guess_atomic = None
    st.session_state.last_guess_atomic = None
    st.session_state.invalid_atomic = None
    st.session_state.win_anim_pending = False
    st.session_state.ui_message = None
    st.session_state.history = []
    st.session_state.prev_hint_set = set()
    st.session_state.stats_recorded = False

    # Force the component to fully remount (prevents "last guess" ghosting)
    _bump_board_nonce()

    if game_mode == "Daily":
        st.session_state.secret = pick_daily(elements)
        st.session_state.seed = None
    else:
        seed = random.randrange(1_000_000_000)
        st.session_state.seed = seed
        st.session_state.secret = pick_endless(elements, seed)


def ensure_state(elements: List[Element]):
    if "game_mode" not in st.session_state:
        st.session_state.game_mode = "Daily"
        st.session_state.difficulty = "Normal"
        st.session_state.enable_special_clue = False  # OFF by default (per request)
        st.session_state.debug_secret_atomic = None
        st.session_state.board_nonce = 0
        start_new_game(elements, st.session_state.game_mode)

    st.session_state.setdefault("enable_special_clue", False)
    st.session_state.setdefault("debug_secret_atomic", None)
    st.session_state.setdefault("ui_message", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("revealed_order", [])
    st.session_state.setdefault("prev_hint_set", set())
    st.session_state.setdefault("invalid_atomic", None)
    st.session_state.setdefault("last_guess_atomic", None)
    st.session_state.setdefault("win_anim_pending", False)
    st.session_state.setdefault("stats_recorded", False)
    st.session_state.setdefault("board_nonce", 0)

    init_stats()


def inject_mobile_css():
    st.markdown(
        """
        <style>
          html, body { overscroll-behavior-y: none; }
          .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# End-game summary (Did you know ABOVE Element Summary)
# =========================================================
def format_K(val: Optional[float]) -> str:
    return "unknown" if val is None else f"{val:.0f} K"


def revealed_property_rows(secret: Element, revealed: Dict[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for k in st.session_state.revealed_order:
        if k not in revealed:
            continue

        if k in ("boil_split", "melt_split"):
            kind = "Boiling point (K)" if k == "boil_split" else "Melting point (K)"
            op, xs = revealed[k].split("|", 1)
            sign = "≤" if op == "le" else ">"
            actual = secret.boiling_point_K if k == "boil_split" else secret.melting_point_K
            rows.append(
                {
                    "Property": kind,
                    "Value": format_K(actual),
                    "Clue": f"{sign} {xs} K",
                    "Revealed by clue": "✅ (partial)",
                }
            )
            continue

        if k == "category3":
            rows.append({"Property": "Category", "Value": secret.category3, "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "state":
            rows.append({"Property": "State at room temp", "Value": secret.state, "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "noble_gas":
            rows.append({"Property": "Noble gas", "Value": "Yes" if secret.is_noble_gas else "No", "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "group_family":
            rows.append({"Property": "Major group family", "Value": group_family(secret), "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "block":
            rows.append({"Property": "Electron block", "Value": block_of(secret), "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "period":
            rows.append({"Property": "Period", "Value": str(secret.period), "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "band":
            rows.append({"Property": "Atomic number range", "Value": atomic_band(secret.atomic_number), "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "radioactive":
            rows.append({"Property": "Radioactive", "Value": "Yes" if is_radioactive(secret) else "No", "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "origin":
            rows.append({"Property": "Origin", "Value": origin_natural_vs_synthetic(secret), "Clue": revealed[k], "Revealed by clue": "✅"})
        elif k == "metal_group":
            rows.append({"Property": "Metal group", "Value": metal_group(secret), "Clue": revealed[k], "Revealed by clue": "✅"})
    return rows


def render_endgame_summary(secret: Element, revealed: Dict[str, str]):
    # Did you know first
    st.subheader("Did you know?")
    bits = []
    if secret.discovered_by:
        bits.append(f"**Discovered by:** {secret.discovered_by}")
    if secret.named_by:
        bits.append(f"**Named by:** {secret.named_by}")
    if bits:
        st.markdown("  \n".join(bits))

    if secret.summary:
        s = secret.summary.strip()
        if len(s) > 650:
            s = s[:650].rsplit(" ", 1)[0] + "…"
        st.write(s)

    if secret.source:
        st.caption(f"Source: {secret.source}")

    # Element Summary second
    st.subheader(f"🧾 Element Summary — {secret.name} ({secret.symbol})")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        if secret.bohr_model_image:
            try:
                st.image(
                    secret.bohr_model_image,
                    caption=f"Bohr model — {secret.name} ({secret.symbol})",
                    use_container_width=True,
                )
            except Exception:
                st.caption("Bohr model image unavailable.")
        else:
            st.caption("No Bohr model image available in dataset.")

    with col2:
        st.markdown(
            f"""
**Atomic number:** {secret.atomic_number}  
**Category:** {secret.category3}  
**Major group family:** {group_family(secret)}  
**Electron block:** {block_of(secret)}  
**State at room temp:** {secret.state}  
**Boiling point:** {format_K(secret.boiling_point_K)}  
**Melting point:** {format_K(secret.melting_point_K)}  
**Radioactive (game rule):** {"Yes" if is_radioactive(secret) else "No"}  
"""
        )

    rows = revealed_property_rows(secret, revealed)
    if rows:
        st.markdown("**What the clues revealed (and/or implied):**")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No clues were revealed in this round (rare).")


# =========================================================
# Main app
# =========================================================
def main():
    st.set_page_config(page_title="Element Guess", page_icon="🧪", layout="wide")
    inject_mobile_css()

    st.title("🧪 Element Guess")
    st.caption("Click an element tile to guess. Colors show your progress.")

    with st.expander("❓ How to play", expanded=False):
        st.markdown(
            """
**Goal:** Guess the hidden element in **7 guesses or fewer**.

**Tile colors:**
- 🟥 **Red** — fails revealed clues
- 🟧 **Amber** — matches clues, but not the answer
- 🟩 **Green** — correct
- 🟦 **Blue** — possible candidates (Easy mode)
- ⬛ **Charcoal** — revealed answer (loss)

**Difficulty:**
- **Easy:** highlights valid candidates
- **Normal:** invalid guesses are locked
- **Hard:** invalid guesses are rejected (no explanation; no guess consumed)

**Special Clue: Atomic Number**
- If enabled, may reveal whether the answer’s atomic number is higher or lower than your last valid guess.
"""
        )

    elements = load_elements()
    by_atomic = {e.atomic_number: e for e in elements}

    ensure_state(elements)

    can_change_settings = (st.session_state.status in ("won", "lost")) or (len(st.session_state.guesses) == 0)

    debug_mode = False

    # ---- Sidebar: order as requested (Restart under Difficulty)
    with st.sidebar:
        st.subheader("Game")

        mode_choice = st.radio(
            "Mode",
            ["Daily", "Endless"],
            index=["Daily", "Endless"].index(st.session_state.game_mode),
            disabled=not can_change_settings,
        )
        if mode_choice != st.session_state.game_mode and can_change_settings:
            st.session_state.game_mode = mode_choice
            start_new_game(elements, st.session_state.game_mode)
            st.rerun()

        st.subheader("Difficulty")
        st.session_state.difficulty = st.radio(
            "Difficulty",
            ["Easy", "Normal", "Hard"],
            index=["Easy", "Normal", "Hard"].index(st.session_state.difficulty)
            if st.session_state.difficulty in ("Easy", "Normal", "Hard")
            else 1,
            disabled=not can_change_settings,
        )

        if st.button("🔄 Restart", use_container_width=True):
            start_new_game(elements, st.session_state.game_mode)
            st.rerun()

        st.subheader("Hints")
        st.session_state.enable_special_clue = st.toggle(
            "Special Clue: Atomic Number",
            value=bool(st.session_state.enable_special_clue),
            help="Reveals whether the answer’s atomic number is higher or lower than your last valid guess (when needed).",
        )

        # Stats in production ✅
        stats_panel()

        # Debug toggle can be enabled in production via ALLOW_DEBUG secret ✅
        if SHOW_DEBUG_UI:
            debug_mode = st.toggle("🛠 Debug mode", value=False)

            if debug_mode:
                st.subheader("🧪 Debug Target Override")
                options = ["— none —"] + [f"{e.name} ({e.symbol}) #{e.atomic_number}" for e in elements]

                current_idx = 0
                if st.session_state.debug_secret_atomic:
                    for i, e in enumerate(elements, start=1):
                        if e.atomic_number == st.session_state.debug_secret_atomic:
                            current_idx = i
                            break

                choice = st.selectbox("Force target element", options, index=current_idx)
                if choice == "— none —":
                    st.session_state.debug_secret_atomic = None
                else:
                    st.session_state.debug_secret_atomic = int(choice.split("#")[-1])

                can_undo = (st.session_state.status == "playing") and (len(st.session_state.history) > 0)
                if st.button("↩️ Undo last valid guess", disabled=not can_undo, use_container_width=True):
                    undo_snapshot()
                    st.session_state.ui_message = "Undid last guess."
                    st.rerun()

                if st.button("Reset stats", use_container_width=True):
                    del st.session_state.stats
                    init_stats()
                    st.success("Stats reset")

    # Daily banner
    if st.session_state.game_mode == "Daily":
        today = dt.date.today()
        st.info(f"📅 Today’s date: {today:%A, %d %B %Y}")

    definitions_panel()

    guessed: Set[int] = set(st.session_state.guesses)

    # Effective secret (debug override does not mutate real secret)
    if debug_mode and st.session_state.debug_secret_atomic:
        secret = by_atomic[st.session_state.debug_secret_atomic]
        st.warning(f"🧪 Debug mode: Target overridden to {secret.name} ({secret.symbol})")
    else:
        secret = st.session_state.secret

    if st.session_state.ui_message:
        st.info(st.session_state.ui_message)
        st.session_state.ui_message = None

    st.write(f"🧩 **Mode:** {st.session_state.game_mode}   |   🎚️ **Difficulty:** {st.session_state.difficulty}")
    st.write(f"🎯 **Guesses used:** {st.session_state.attempt}/{st.session_state.max_guesses}")

    guesses_left = st.session_state.max_guesses - st.session_state.attempt
    special_atomic = compute_special_atomic_clue(
        enabled=bool(st.session_state.enable_special_clue),
        elements=elements,
        revealed=st.session_state.revealed,
        guessed=guessed,
        guesses_left=guesses_left,
        secret=secret,
        last_valid_guess_atomic=st.session_state.last_valid_guess_atomic,
    )

    if debug_mode:
        with st.expander("🛠 Debug analytics", expanded=True):
            show_debug_panel(secret, elements, st.session_state.revealed)

    invalid_atomic_to_send = st.session_state.invalid_atomic
    last_guess_to_send = st.session_state.last_guess_atomic
    win_anim_pending = bool(st.session_state.win_anim_pending)

    tiles = make_tiles(
        elements=elements,
        revealed=st.session_state.revealed,
        guessed=guessed,
        secret_atomic=secret.atomic_number,
        difficulty=st.session_state.difficulty,
        last_guess_atomic=last_guess_to_send,
        win_anim_pending=win_anim_pending,
        status=st.session_state.status,
    )

    # IMPORTANT: key includes board_nonce to prevent "ghost last guess" after restart/mode switch
    board_key = f"tiles_board_{st.session_state.board_nonce}"

    click = periodic_table(
        tiles=tiles,
        legend={
            "none": "#E5E7EB",
            "bad": "#EF4444",
            "close": "#F59E0B",
            "correct": "#16A34A",
            "hint": "#3B82F6",
            "lost": "#111827",
        },
        disabled=(st.session_state.status != "playing"),
        invalidAtomic=invalid_atomic_to_send,
        key=board_key,
    )

    # Clear one-shot triggers after render
    if invalid_atomic_to_send is not None:
        st.session_state.invalid_atomic = None
    if last_guess_to_send is not None:
        st.session_state.last_guess_atomic = None
    if win_anim_pending:
        st.session_state.win_anim_pending = False

    # Handle clicks
    if click and st.session_state.status == "playing":
        atomic = click.get("atomic")
        nonce = click.get("nonce")

        if nonce is not None and nonce == st.session_state.last_click_nonce:
            pass
        else:
            st.session_state.last_click_nonce = nonce

            if not isinstance(atomic, int) or atomic not in by_atomic:
                st.session_state.ui_message = "Received an invalid click from the board."
                st.rerun()

            e = by_atomic[atomic]

            # Normal + Hard: invalid guess doesn't consume a guess; shake + message
            if st.session_state.revealed and (not matches(e, st.session_state.revealed)):
                if st.session_state.difficulty in ("Normal", "Hard"):
                    st.session_state.invalid_atomic = atomic
                    st.session_state.ui_message = "Not a valid choice — try a different element."
                    st.rerun()

            if atomic in guessed:
                st.session_state.ui_message = f"You already guessed {e.name} ({e.symbol})."
                st.rerun()

            if debug_mode:
                push_snapshot()

            st.session_state.last_valid_guess_atomic = atomic
            st.session_state.last_guess_atomic = atomic

            if atomic == secret.atomic_number:
                feedback = "correct"
            else:
                feedback = "close" if matches(e, st.session_state.revealed) else "bad"

            st.session_state.guesses.append(atomic)
            st.session_state.guess_feedback.append(feedback)
            st.session_state.attempt += 1

            if atomic == secret.atomic_number:
                st.session_state.status = "won"
                st.session_state.win_anim_pending = True
                st.rerun()

            # Reveal next clue using entropy (skips useless clues)
            reveal_next_clue(secret, elements, st.session_state.attempt)

            if st.session_state.attempt >= st.session_state.max_guesses:
                st.session_state.status = "lost"

            st.rerun()

    # Revealed clues — numbered, chronological
    if st.session_state.revealed or special_atomic:
        st.subheader("Revealed clues")
        i = 1
        for k in st.session_state.revealed_order:
            if k not in st.session_state.revealed:
                continue
            v = st.session_state.revealed[k]
            if k in ("boil_split", "melt_split"):
                op, xs = v.split("|", 1)
                sign = "≤" if op == "le" else ">"
                st.write(f"{i}. **{CLUES[k]}:** {sign} {xs} K")
            else:
                st.write(f"{i}. **{CLUES[k]}:** {v}")
            i += 1

        if special_atomic:
            st.write(f"{i}. **Special clue:** {special_atomic}")

    if st.session_state.guesses:
        st.subheader("Guesses")
        labels = [f"{by_atomic[a].name} ({by_atomic[a].symbol})" for a in st.session_state.guesses]
        st.write(" • ".join(labels))

    st.subheader("🔗 Share this game")
    copy_link_button()
    st.caption("Tip: You can also copy the URL from your browser’s address bar.")

    # Record stats exactly once on game end
    if st.session_state.status in ("won", "lost") and not st.session_state.stats_recorded:
        if st.session_state.status == "won":
            record_win(len(st.session_state.guesses))
        else:
            record_loss()
        st.session_state.stats_recorded = True

    # End states + summary + share results
    if st.session_state.status in ("won", "lost"):
        if st.session_state.status == "won":
            st.success(f"🎉 Correct! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")
        else:
            st.error(f"😅 Out of guesses! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")
            st.caption("⬛ The answer has been revealed on the periodic table.")

        render_endgame_summary(secret, st.session_state.revealed)

        st.subheader("📋 Share results")
        share_text = build_share_text()
        copy_to_clipboard_button(share_text, label="📋 Copy results")
        with st.expander("Preview share text", expanded=False):
            st.code(share_text)
    else:
        st.caption("Finish the game to unlock the Element Summary + Share Results.")

    st.markdown("---")
    st.caption(f"🧪 Element Guess • {APP_VERSION} • GitHub: {GITHUB_URL}")


if __name__ == "__main__":
    main()
