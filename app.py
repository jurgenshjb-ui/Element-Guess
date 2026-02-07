from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import urllib.parse

import streamlit as st
import streamlit.components.v1 as components

# deploy bump: local_storage assets path fix

# deploy bump

# Your custom periodic-table component
from element_tiles import periodic_table

# =========================================================
# App metadata (edit to match your repo)
# =========================================================
APP_VERSION = "v1.4.0"
GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO"
WIKI_BASE = "https://en.wikipedia.org/wiki/"
RSC_SEARCH = "https://www.rsc.org/periodic-table/element/"  # slug by atomic number works
WEBELEMENTS_BASE = "https://www.webelements.com/"          # symbol path is typical, but varies

# =========================================================
# Production / Debug gating
# Streamlit Cloud -> App settings -> Secrets:
# STREAMLIT_ENV = "prod"
# DEBUG = "false"   (or "true" to enable debug UI)
# =========================================================
IS_PRODUCTION = (
    os.environ.get("STREAMLIT_ENV") == "prod"
    or st.secrets.get("STREAMLIT_ENV", "dev") == "prod"
)
DEBUG_ALLOWED = str(st.secrets.get("DEBUG", "false")).lower() == "true"
SHOW_DEBUG_UI = (not IS_PRODUCTION) and DEBUG_ALLOWED or (not IS_PRODUCTION and not IS_PRODUCTION)

# If you want debug hidden in prod but available locally:
# - set STREAMLIT_ENV="prod" in Secrets to hide debug on Cloud
# - keep DEBUG="false" (or omit) on Cloud



# =========================================================
# LocalStorage stats component (optional)
# ---------------------------------------------------------
# This app will TRY to use a local component if you create it.
# If the component isn't present, it falls back to session-only stats.
#
# To enable per-device stats persistence:
# - Create folder: local_storage/frontend/dist
# - Build a simple Streamlit component that reads/writes localStorage
#
# You can also keep using session stats; the app will still run.
# =========================================================
def _declare_local_storage_component():
    build_dir = os.path.join(
        os.path.dirname(__file__),
        "local_storage",
        "frontend",
        "dist"
    )

    if not os.path.exists(build_dir):
        st.warning("⚠️ local_storage component not found — stats will be session-only")
        return None

    return components.declare_component("local_storage", path=build_dir)



_LOCAL_STORAGE_COMPONENT = _declare_local_storage_component()


def _html_storage_get(key: str) -> Optional[dict]:
    # Use a unique marker in the URL query params to receive JS response
    marker = f"ls_{key}"
    q = st.query_params

    # If JS already wrote it into query params, read it
    if marker in q:
        try:
            raw = q.get(marker)
            if raw is None:
                return None
            data = json.loads(urllib.parse.unquote(raw))
            # Important: don't leave it in URL forever
            # Remove only our marker key
            try:
                del st.query_params[marker]
            except Exception:
                pass
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    # Otherwise inject JS that reads localStorage and writes it into query params
    js = f"""
    <script>
      (function() {{
        try {{
          const key = {json.dumps(key)};
          const marker = {json.dumps(marker)};
          const raw = window.localStorage.getItem(key);
          const val = raw ? raw : "null";
          const url = new URL(window.location.href);
          url.searchParams.set(marker, encodeURIComponent(val));
          window.history.replaceState(null, "", url.toString());
          // Trigger Streamlit rerun by posting a message
          window.parent.postMessage({{ type: "streamlit:rerun" }}, "*");
        }} catch (e) {{}}
      }})();
    </script>
    """
    components.html(js, height=0)
    return None


def _html_storage_set(key: str, value: dict) -> None:
    js = f"""
    <script>
      (function() {{
        try {{
          const key = {json.dumps(key)};
          const value = {json.dumps(json.dumps(value))}; // stringify safely
          window.localStorage.setItem(key, value);
        }} catch (e) {{}}
      }})();
    </script>
    """
    components.html(js, height=0)


def local_storage_get(key: str) -> Optional[dict]:
    if _LOCAL_STORAGE_COMPONENT is None:
        return None
    try:
        out = _LOCAL_STORAGE_COMPONENT(op="get", key=key, default=None, key=f"ls_get_{key}")
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def local_storage_set(key: str, value: dict) -> None:
    if _LOCAL_STORAGE_COMPONENT is None:
        return
    try:
        _LOCAL_STORAGE_COMPONENT(op="set", key=key, value=value, default=None, key=f"ls_set_{key}")
    except Exception:
        pass


    # 2) Fallback
    _html_storage_set(key, value)

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
    # Rough, but consistent and useful: mostly synthetics >=93, plus Tc/Pm
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
# Clues + ordering
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

# Used only as a tie-breaker when multiple clues leave the same pool size
CLUE_ORDER = [
    "category3",
    "group_family",
    "block",
    "band",
    "state",        # phase before noble gas
    "noble_gas",    # only eligible if state == gas
    "period",
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


# =========================================================
# Temperature split helpers
# - X chosen ~median of known vals, rounded to nearest 100 K
# - Only eligible from guess >= 3
# - Only one of boil/melt appears per game (mutual exclusion)
# - Require at least 70% known among current candidates to avoid "unknown" chaos
# =========================================================
def _choose_temp_split_threshold(current_candidates: List[Element], kind: str) -> Optional[float]:
    vals = [(_temp_value_K(e, kind)) for e in current_candidates]
    known_vals = sorted([v for v in vals if v is not None])
    total = len(current_candidates)
    known = len(known_vals)

    if total == 0 or known < 6:
        return None
    if (known / total) < 0.70:
        return None

    median = known_vals[len(known_vals) // 2]
    x = round(median / 100.0) * 100.0
    return float(x)


def _secret_bucket_for_temp(secret: Element, clue_key: str, x: float) -> Optional[str]:
    kind = "boil" if clue_key == "boil_split" else "melt"
    val = _temp_value_K(secret, kind)
    if val is None:
        return None
    return "≤X" if val <= x else ">X"


def _count_candidates_for_secret_temp_value(
    secret: Element,
    current_candidates: List[Element],
    clue_key: str,
    x: float,
) -> Optional[Tuple[str, int]]:
    bucket = _secret_bucket_for_temp(secret, clue_key, x)
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


def _deterministic_tiebreak(game_mode: str, seed: Optional[int]) -> int:
    """
    Returns 0 or 1 deterministically per game.
    Daily: based on date ordinal. Endless: based on seed.
    Used only when boil vs melt tie perfectly.
    """
    if game_mode == "Daily":
        return dt.date.today().toordinal() % 2
    return (seed or 0) % 2


# =========================================================
# Period gating (anti-spoiler)
# Only allow Period if:
# - attempt >= 4
# - remaining candidates > 18
# - and it narrows (checked in selection)
# =========================================================
def _period_is_eligible(attempt: int, current_n: int) -> bool:
    return (attempt >= 4) and (current_n > 18)


# =========================================================
# Allowed clue keys (gameplay)
# =========================================================
def allowed_clue_keys(revealed: Dict[str, str], attempt: int, current_n: int) -> List[str]:
    keys = [k for k in CLUE_ORDER if k not in revealed]

    # radioactive only after guess 3
    if attempt < 3 and "radioactive" in keys:
        keys.remove("radioactive")

    # temperature splits only after guess 3
    if attempt < 3:
        if "boil_split" in keys:
            keys.remove("boil_split")
        if "melt_split" in keys:
            keys.remove("melt_split")

    # only one temperature clue per game
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

    # period gating
    if "period" in keys and (not _period_is_eligible(attempt, current_n)):
        keys.remove("period")

    return keys


def debug_unrevealed_keys(revealed: Dict[str, str]) -> List[str]:
    # Debug ignores gameplay gating, but keeps mutual exclusion of boil/melt
    keys = [k for k in CLUE_ORDER if k not in revealed]
    if "boil_split" in revealed and "melt_split" in keys:
        keys.remove("melt_split")
    if "melt_split" in revealed and "boil_split" in keys:
        keys.remove("boil_split")
    return keys


# =========================================================
# Previous entropy-free clue selection:
# Choose the clue that leaves the WIDEST remaining pool (for secret’s value),
# but skip useless clues that don't narrow (cnt == current_n) or eliminate (cnt == 0).
# =========================================================
def choose_next_clue_max_pool(
    secret: Element,
    elements: List[Element],
    revealed: Dict[str, str],
    attempt: int,
    game_mode: str,
    seed: Optional[int],
) -> Optional[Tuple[str, str, int]]:
    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    if current_n <= 1:
        return None

    remaining_keys = allowed_clue_keys(revealed, attempt, current_n)
    if not remaining_keys:
        return None

    # Precompute thresholds for temp splits
    boil_x = _choose_temp_split_threshold(current_candidates, "boil") if "boil_split" in remaining_keys else None
    melt_x = _choose_temp_split_threshold(current_candidates, "melt") if "melt_split" in remaining_keys else None

    best: Optional[Tuple[str, str, int, int, int]] = None
    # (key, value, cnt_secret, tie_rank, order_index)

    for k in remaining_keys:
        if k in ("boil_split", "melt_split"):
            x = boil_x if k == "boil_split" else melt_x
            if x is None:
                continue
            res = _count_candidates_for_secret_temp_value(secret, current_candidates, k, x)
            if res is None:
                continue
            bucket, cnt = res
            if not (0 < cnt < current_n):
                continue

            op = "le" if bucket == "≤X" else "gt"
            v = f"{op}|{int(x)}"

            temp_rank = _deterministic_tiebreak(game_mode, seed)
            prefer_melt = (temp_rank == 1)
            this_is_melt = (k == "melt_split")
            tie_rank = 0 if (this_is_melt == prefer_melt) else 1

            cand = (k, v, cnt, tie_rank, CLUE_ORDER.index(k))
        else:
            secret_value = prop(secret, k)
            cnt = sum(1 for e in current_candidates if prop(e, k) == secret_value)
            if not (0 < cnt < current_n):
                continue
            cand = (k, secret_value, cnt, 0, CLUE_ORDER.index(k))

        if best is None:
            best = cand
        else:
            _, _, best_cnt, best_tie, best_ord = best
            _, _, cnt, tie_rank, ord_idx = cand

            if cnt > best_cnt:
                best = cand
            elif cnt == best_cnt:
                if ord_idx < best_ord:
                    best = cand
                elif ord_idx == best_ord:
                    if tie_rank < best_tie:
                        best = cand

    if best is None:
        return None

    k, v, cnt, _, _ = best
    return k, v, cnt


def reveal_next_clue(secret: Element, elements: List[Element], attempt: int) -> None:
    nxt = choose_next_clue_max_pool(
        secret=secret,
        elements=elements,
        revealed=st.session_state.revealed,
        attempt=attempt,
        game_mode=st.session_state.game_mode,
        seed=st.session_state.seed,
    )
    if not nxt:
        return
    k, v, _ = nxt
    if k not in st.session_state.revealed:
        st.session_state.revealed[k] = v
        st.session_state.revealed_order.append(k)
        # Track candidates after each clue for testing metrics
        _record_candidates_checkpoint(elements, secret)


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
# Daily / Endless selection + share metadata
# =========================================================
DAILY_EPOCH = dt.date(2026, 1, 1)


def day_number(date: Optional[dt.date] = None) -> int:
    date = date or dt.date.today()
    return abs((date - DAILY_EPOCH).days) + 1


def daily_fingerprint(secret_atomic: int, date: Optional[dt.date] = None) -> str:
    date = date or dt.date.today()
    s = f"{date.isoformat()}|{secret_atomic}|element-guess"
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:4].upper()


def pick_daily(elements: List[Element], date: Optional[dt.date] = None) -> Element:
    date = date or dt.date.today()
    idx = abs((date - DAILY_EPOCH).days) % len(elements)
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
# Debug distributions
# =========================================================
def _distribution_table(current_candidates: List[Element], clue_key: str) -> List[dict]:
    counts: Dict[str, int] = {}

    if clue_key in ("boil_split", "melt_split"):
        kind = "boil" if clue_key == "boil_split" else "melt"
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

    # Impact: unrevealed clue -> candidates if revealed secret's value
    rows = []

    # Precompute thresholds for temp splits (debug can show even if gameplay-gated)
    boil_x = _choose_temp_split_threshold(current_candidates, "boil")
    melt_x = _choose_temp_split_threshold(current_candidates, "melt")

    for k in debug_unrevealed_keys(revealed):
        if k in ("boil_split", "melt_split"):
            x = boil_x if k == "boil_split" else melt_x
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
# Tile placement
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


# =========================================================
# Game state + metrics checkpoints for testing
# =========================================================
def _candidate_count(elements: List[Element], revealed: Dict[str, str]) -> int:
    return sum(1 for e in elements if matches(e, revealed))


def _init_candidate_history(elements: List[Element]):
    # Track candidates after each clue reveal, starting at initial state
    st.session_state.candidate_history = [_candidate_count(elements, st.session_state.revealed)]


def _record_candidates_checkpoint(elements: List[Element], secret: Element):
    # Called after a new clue is revealed
    if "candidate_history" not in st.session_state:
        st.session_state.candidate_history = []
    st.session_state.candidate_history.append(_candidate_count(elements, st.session_state.revealed))


# =========================================================
# Test harness (debug-only): curated targets + last-3 run metrics
# =========================================================
DEFAULT_TEST_TARGETS = [
    1,   # H
    10,  # Ne
    35,  # Br
    11,  # Na
    12,  # Mg
    26,  # Fe
    29,  # Cu
    80,  # Hg
    14,  # Si
    6,   # C
    17,  # Cl
    53,  # I
    60,  # Nd
    92,  # U
    95,  # Am
]


def _parse_targets_text(text: str, by_atomic: Dict[int, Element]) -> List[int]:
    """
    Accept comma-separated atomic numbers, symbols, or names.
    Examples:
      "Br, 35, Neon, U"
    """
    out: List[int] = []
    tokens = [t.strip() for t in (text or "").split(",") if t.strip()]
    if not tokens:
        return DEFAULT_TEST_TARGETS.copy()

    # reverse lookup
    by_sym = {e.symbol.lower(): e.atomic_number for e in by_atomic.values()}
    by_name = {e.name.lower(): e.atomic_number for e in by_atomic.values()}

    for tok in tokens:
        low = tok.lower()
        if low.isdigit():
            n = int(low)
            if n in by_atomic:
                out.append(n)
            continue
        if low in by_sym:
            out.append(by_sym[low])
            continue
        if low in by_name:
            out.append(by_name[low])
            continue
    # de-dupe while keeping order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq if uniq else DEFAULT_TEST_TARGETS.copy()


def _ensure_test_metrics_store():
    """
    Store last 3 game records per atomic.
    record = {
      "ts": "...",
      "mode": "Daily/Endless",
      "difficulty": "...",
      "result": "won/lost",
      "guesses_used": int,
      "candidate_history": [int, int, ...],  # initial + after each clue
    }
    """
    if "test_metrics" not in st.session_state:
        st.session_state.test_metrics = {}  # atomic -> list[record]


def _record_test_run(secret_atomic: int):
    if "debug_enabled" not in st.session_state or not st.session_state.debug_enabled:
        return
    if not st.session_state.get("debug_target_override_atomic"):
        return

    _ensure_test_metrics_store()
    rec = {
        "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": st.session_state.game_mode,
        "difficulty": st.session_state.difficulty,
        "result": st.session_state.status,
        "guesses_used": len(st.session_state.guesses),
        "candidate_history": list(st.session_state.get("candidate_history", [])),
    }
    arr = st.session_state.test_metrics.get(secret_atomic, [])
    arr.append(rec)
    arr = arr[-3:]  # keep last 3
    st.session_state.test_metrics[secret_atomic] = arr


def _render_test_metrics(by_atomic: Dict[int, Element], targets: List[int]):
    _ensure_test_metrics_store()
    rows = []
    for n in targets:
        e = by_atomic.get(n)
        if not e:
            continue
        runs = st.session_state.test_metrics.get(n, [])
        if not runs:
            rows.append({
                "Target": f"{e.name} ({e.symbol}) #{n}",
                "Last 3 win rate": "—",
                "Avg guesses (last 3)": "—",
                "Avg candidates after each clue (last 3)": "—",
            })
            continue

        wins = sum(1 for r in runs if r["result"] == "won")
        win_rate = wins / len(runs) * 100.0
        avg_guesses = sum(r["guesses_used"] for r in runs) / len(runs)

        # average candidate history by index
        max_len = max((len(r.get("candidate_history", [])) for r in runs), default=0)
        avg_hist = []
        for i in range(max_len):
            vals = []
            for r in runs:
                h = r.get("candidate_history", [])
                if i < len(h):
                    vals.append(h[i])
            if vals:
                avg_hist.append(int(round(sum(vals) / len(vals))))
        avg_hist_str = " → ".join(str(x) for x in avg_hist) if avg_hist else "—"

        rows.append({
            "Target": f"{e.name} ({e.symbol}) #{n}",
            "Last 3 win rate": f"{win_rate:.0f}%",
            "Avg guesses (last 3)": f"{avg_guesses:.2f}",
            "Avg candidates after each clue (last 3)": avg_hist_str,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


# =========================================================
# Tiles + statuses + animations
# =========================================================
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
        # reveal secret on loss if not already guessed
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


def build_share_text(secret: Element) -> str:
    mode = st.session_state.game_mode
    difficulty = st.session_state.difficulty
    guesses_used = len(st.session_state.guesses)
    max_guesses = st.session_state.max_guesses
    rows = [emoji_for_guess(x) for x in st.session_state.guess_feedback]
    grid = "\n".join(rows) if rows else ""

    if mode == "Daily":
        dnum = day_number()
        fp = daily_fingerprint(secret.atomic_number)
        header = f"Element Guess — Day {dnum} ({difficulty})"
        meta = f"Mode: Daily • Fingerprint: {fp}"
    else:
        header = f"Element Guess — Endless ({difficulty})"
        meta = f"Mode: Endless"

    score = f"Score: {guesses_used}/{max_guesses}"
    return f"{header}\n{score}\n{meta}\n\n{grid}".strip()


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
# Stats (per-device if localStorage component exists; else session)
# =========================================================
STATS_KEY = "element_guess_stats_v1"


def _default_stats() -> dict:
    return {
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "current_streak": 0,
        "best_streak": 0,
        "total_guesses_in_wins": 0,
        "win_guess_counts": [],
    }


def load_stats() -> dict:
    # We allow up to 2 hydration attempts because components return default on first run.
    if "_stats_hydrate_attempts" not in st.session_state:
        st.session_state._stats_hydrate_attempts = 0

    if "stats" not in st.session_state or not isinstance(st.session_state.stats, dict):
        st.session_state.stats = _default_stats()

    # If we haven't successfully hydrated yet, keep trying (up to 2 attempts)
    if not st.session_state.get("_stats_hydrated", False):
        st.session_state._stats_hydrate_attempts += 1

        persisted = local_storage_get(STATS_KEY)
        if isinstance(persisted, dict) and persisted.get("games_played") is not None:
            st.session_state.stats = persisted
            st.session_state._stats_hydrated = True
        elif st.session_state._stats_hydrate_attempts >= 2:
            # After 2 tries, accept "no stored stats" and stop retrying
            st.session_state._stats_hydrated = True

    return st.session_state.stats



def save_stats(stats: dict) -> None:
    st.session_state.stats = stats
    local_storage_set(STATS_KEY, stats)


def record_win(stats: dict, guesses_used: int) -> dict:
    stats["games_played"] += 1
    stats["wins"] += 1
    stats["current_streak"] += 1
    stats["best_streak"] = max(stats["best_streak"], stats["current_streak"])
    stats["total_guesses_in_wins"] += guesses_used
    stats["win_guess_counts"].append(int(guesses_used))
    return stats


def record_loss(stats: dict) -> dict:
    stats["games_played"] += 1
    stats["losses"] += 1
    stats["current_streak"] = 0
    return stats


def stats_panel():
    stats = load_stats()
    with st.expander("📊 Stats", expanded=False):
        if stats["games_played"] == 0:
            st.caption("No games played yet.")
            return

        win_rate = (stats["wins"] / stats["games_played"]) * 100
        st.markdown(
            f"""
**Games played:** {stats["games_played"]}  
**Wins:** {stats["wins"]}  
**Losses:** {stats["losses"]}  
**Win rate:** {win_rate:.1f}%  

**Current streak:** {stats["current_streak"]}  
**Best streak:** {stats["best_streak"]}  
"""
        )

        if stats["wins"] > 0:
            avg_guesses = stats["total_guesses_in_wins"] / stats["wins"]
            st.markdown(f"**Average guesses (wins):** {avg_guesses:.2f}")

            dist: Dict[int, int] = {}
            for g in stats["win_guess_counts"]:
                dist[g] = dist.get(g, 0) + 1

            st.markdown("**Guess distribution (wins):**")
            for g in sorted(dist):
                st.markdown(f"- {g} guesses: {dist[g]}")


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
        candidate_history=list(st.session_state.get("candidate_history", [])),
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
    st.session_state.candidate_history = snap.get("candidate_history", [])


# =========================================================
# UI helpers
# =========================================================
def inject_mobile_css():
    st.markdown(
        """
        <style>
          html, body { overscroll-behavior-y: none; }
          .block-container { padding-top: 1rem; padding-bottom: 2rem; }
          /* Optional: slightly larger click targets on mobile */
          @media (max-width: 768px) {
            button[kind="primary"] { padding: 0.6rem 0.9rem !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _bump_board_nonce():
    st.session_state.board_nonce = int(st.session_state.get("board_nonce", 0)) + 1


# =========================================================
# Game lifecycle
# =========================================================
def start_new_game(elements: List[Element], game_mode: str):
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

    # Force component remount (prevents "ghost last guess" on mode swap/restart)
    _bump_board_nonce()

    if game_mode == "Daily":
        st.session_state.secret = pick_daily(elements)
        st.session_state.seed = None
    else:
        seed = random.randrange(1_000_000_000)
        st.session_state.seed = seed
        st.session_state.secret = pick_endless(elements, seed)

    _init_candidate_history(elements)


def ensure_state(elements: List[Element]):
    if "game_mode" not in st.session_state:
        st.session_state.game_mode = "Daily"
        st.session_state.difficulty = "Normal"
        st.session_state.enable_special_clue = False  # OFF by default
        st.session_state.debug_enabled = False
        st.session_state.debug_target_override_atomic = None
        st.session_state.board_nonce = 0
        start_new_game(elements, st.session_state.game_mode)

    st.session_state.setdefault("enable_special_clue", False)
    st.session_state.setdefault("debug_enabled", False)
    st.session_state.setdefault("debug_target_override_atomic", None)
    st.session_state.setdefault("ui_message", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("revealed_order", [])
    st.session_state.setdefault("prev_hint_set", set())
    st.session_state.setdefault("invalid_atomic", None)
    st.session_state.setdefault("last_guess_atomic", None)
    st.session_state.setdefault("win_anim_pending", False)
    st.session_state.setdefault("stats_recorded", False)
    st.session_state.setdefault("board_nonce", 0)
    st.session_state.setdefault("candidate_history", [])
    load_stats()  # initialize stats in state


# =========================================================
# End-game summary: "Game-first" facts grid
# - show properties that are current or future clue candidates
# - show ✅ revealed / ❌ not revealed
# =========================================================
# Which properties are eligible for clues now or in future:
CLUE_ELIGIBLE_PROPS = [
    "category3",
    "group_family",
    "block",
    "band",
    "state",
    "noble_gas",
    "radioactive",
    "origin",
    "metal_group",
    "period",      # still eligible, but gated
    "boil_point",  # future/aux info (we reveal split, but show actual too)
    "melt_point",  # future/aux info (we reveal split, but show actual too)
]


def format_K(val: Optional[float]) -> str:
    return "unknown" if val is None else f"{val:.0f} K"


def _prop_value_for_summary(secret: Element, p: str) -> str:
    if p == "category3":
        return secret.category3
    if p == "group_family":
        return group_family(secret)
    if p == "block":
        return block_of(secret)
    if p == "band":
        return atomic_band(secret.atomic_number)
    if p == "state":
        return secret.state
    if p == "noble_gas":
        return "Yes" if secret.is_noble_gas else "No"
    if p == "radioactive":
        return "Yes" if is_radioactive(secret) else "No"
    if p == "origin":
        return origin_natural_vs_synthetic(secret)
    if p == "metal_group":
        return metal_group(secret)
    if p == "period":
        return str(secret.period)
    if p == "boil_point":
        return format_K(secret.boiling_point_K)
    if p == "melt_point":
        return format_K(secret.melting_point_K)
    return "unknown"


def _display_name_for_summary_prop(p: str) -> str:
    return {
        "category3": "Category",
        "group_family": "Major group family",
        "block": "Electron block",
        "band": "Atomic number range",
        "state": "State at room temp",
        "noble_gas": "Noble gas",
        "radioactive": "Radioactive",
        "origin": "Origin (natural/synthetic)",
        "metal_group": "Metal group",
        "period": "Period",
        "boil_point": "Boiling point (K)",
        "melt_point": "Melting point (K)",
    }.get(p, p)


def _revealed_status_for_summary_prop(p: str, revealed: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (pill_text, pill_emoji) for this property.
    If the property corresponds to a clue key, it's revealed iff that clue was revealed.
    For boil/melt actual values: revealed iff the split clue of same family was revealed.
    """
    # direct clue keys
    if p in ("category3", "group_family", "block", "band", "state", "noble_gas", "radioactive", "origin", "metal_group", "period"):
        key = "group_family" if p == "group_family" else p
        revealed_flag = key in revealed
        return ("Revealed", "✅") if revealed_flag else ("Not revealed", "❌")

    if p == "boil_point":
        return ("Revealed", "✅") if ("boil_split" in revealed) else ("Not revealed", "❌")
    if p == "melt_point":
        return ("Revealed", "✅") if ("melt_split" in revealed) else ("Not revealed", "❌")

    return ("Not revealed", "❌")


def _facts_rows(secret: Element, revealed: Dict[str, str], group: str) -> List[dict]:
    """
    group: "core" | "thermal" | "advanced"
    """
    if group == "core":
        props = ["category3", "group_family", "block", "state", "band", "noble_gas"]
    elif group == "thermal":
        props = ["melt_point", "boil_point"]
    else:
        # Advanced / conditional
        props = ["radioactive", "origin", "metal_group", "period"]

    rows = []
    for p in props:
        # Keep noble gas meaningful: it's only useful when state==gas, but we can still show it.
        value = _prop_value_for_summary(secret, p)
        pill, emoji = _revealed_status_for_summary_prop(p, revealed)
        rows.append({
            "Property": _display_name_for_summary_prop(p),
            "Value": value,
            "Status": f"{emoji} {pill}",
        })
    return rows


def _external_links(secret: Element) -> List[Tuple[str, str]]:
    # Wikipedia uses element name typically
    wiki = WIKI_BASE + secret.name.replace(" ", "_")
    # RSC: element/atomic_number
    rsc = RSC_SEARCH + str(secret.atomic_number)
    # WebElements: often /symbol/ - keep best-effort
    we = WEBELEMENTS_BASE + secret.symbol.lower() + ".html"
    return [("Wikipedia", wiki), ("RSC", rsc), ("WebElements", we)]


def render_endgame_summary(secret: Element, revealed: Dict[str, str], status: str):
    # Did you know (top)
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

    # Element Summary Card
    st.subheader(f"🧾 Element Summary — {secret.name} ({secret.symbol})")

    # Simple revealed ratio for clue-eligible props (excluding thermal actuals)
    total_props = 0
    revealed_props = 0
    for p in CLUE_ELIGIBLE_PROPS:
        if p in ("boil_point", "melt_point"):
            continue
        total_props += 1
        pill, emoji = _revealed_status_for_summary_prop(p, revealed)
        if emoji == "✅":
            revealed_props += 1

    st.caption(f"Clues revealed: {revealed_props} / {total_props} properties ✅")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        # Enlarge tile effect: we render a mini "single-tile" card using HTML for clarity
        tile_bg = "#16A34A" if status == "won" else "#111827"
        txt = "#FFFFFF" if status == "won" else "#FEF9C3"
        subtxt = "rgba(255,255,255,0.75)" if status != "won" else "rgba(255,255,255,0.85)"
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:center; gap:10px;">
              <div style="
                width:120px; height:120px; border-radius:18px;
                border:1px solid rgba(0,0,0,0.18);
                background:{tile_bg};
                box-shadow: 0 2px 0 rgba(0,0,0,0.06), 0 0 0 1px rgba(255,255,255,0.08) inset;
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                font-family:system-ui; user-select:none;">
                <div style="font-size:16px; font-weight:700; color:{subtxt}; line-height:16px;">{secret.atomic_number}</div>
                <div style="font-size:42px; font-weight:900; color:{txt}; line-height:44px; letter-spacing:0.4px;">{secret.symbol}</div>
              </div>
              <div style="font-family:system-ui; font-weight:700; color:#374151;">
                {"Correct!" if status=="won" else "Answer revealed"}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if secret.bohr_model_image:
            try:
                st.image(secret.bohr_model_image, caption="Bohr model", use_container_width=True)
            except Exception:
                st.caption("Bohr model image unavailable.")

    with col2:
        # Core facts (expanded)
        with st.expander("Core identity", expanded=True):
            st.dataframe(_facts_rows(secret, revealed, "core"), use_container_width=True, hide_index=True)

        with st.expander("Thermal properties", expanded=False):
            st.dataframe(_facts_rows(secret, revealed, "thermal"), use_container_width=True, hide_index=True)
            # If a split clue was revealed, show it explicitly as well (nice clarity)
            if "boil_split" in revealed:
                op, xs = revealed["boil_split"].split("|", 1)
                sign = "≤" if op == "le" else ">"
                st.caption(f"Boiling split clue used: {sign} {xs} K")
            if "melt_split" in revealed:
                op, xs = revealed["melt_split"].split("|", 1)
                sign = "≤" if op == "le" else ">"
                st.caption(f"Melting split clue used: {sign} {xs} K")

        with st.expander("Advanced / conditional", expanded=False):
            st.dataframe(_facts_rows(secret, revealed, "advanced"), use_container_width=True, hide_index=True)

        # External links
        st.markdown("**Learn more:**")
        links = _external_links(secret)
        cols = st.columns(len(links))
        for i, (label, url) in enumerate(links):
            cols[i].link_button(label, url)


# =========================================================
# How to play section
# =========================================================
def how_to_play():
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

**Hints**
- **Special Clue: Atomic Number** can reveal whether the answer’s atomic number is higher or lower than your last valid guess (when needed).
"""
        )


# =========================================================
# Main
# =========================================================
def main():
    st.set_page_config(page_title="Element Guess", page_icon="🧪", layout="wide")
    inject_mobile_css()

    st.title("🧪 Element Guess")
    st.caption("Click an element tile to guess. Colors show your progress.")

    how_to_play()

    elements = load_elements()
    by_atomic = {e.atomic_number: e for e in elements}

    ensure_state(elements)

    # Determine whether settings can change
    can_change_settings = (st.session_state.status in ("won", "lost")) or (len(st.session_state.guesses) == 0)

    # Daily banner
    if st.session_state.game_mode == "Daily":
        today = dt.date.today()
        st.info(f"📅 Today: {today:%A, %d %B %Y} • Day {day_number()}")

    definitions_panel()

    # Effective secret (debug override does not mutate real secret)
    debug_mode = False
    if SHOW_DEBUG_UI:
        # We'll toggle debug via sidebar, but keep default false
        pass

    # ---- Sidebar
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
            help="May reveal whether the answer’s atomic number is higher or lower than your last valid guess (when needed).",
        )

        stats_panel()

        # Debug UI (hidden in prod by setting STREAMLIT_ENV=prod in secrets)
        if SHOW_DEBUG_UI:
            st.divider()
            debug_mode = st.toggle("🛠 Debug mode", value=False)
            st.session_state.debug_enabled = bool(debug_mode)

            if debug_mode:
                st.subheader("🧪 Test Targets")

                # Hot-swappable text list
                default_text = ", ".join(str(n) for n in st.session_state.get("test_targets", DEFAULT_TEST_TARGETS))
                text = st.text_area(
                    "Targets (comma-separated atomic #, symbol, or name)",
                    value=default_text,
                    height=60,
                    help="Example: Br, 35, Neon, U",
                )
                targets = _parse_targets_text(text, by_atomic)
                st.session_state.test_targets = targets

                # Dropdown override
                options = ["— none —"] + [f"{by_atomic[n].name} ({by_atomic[n].symbol}) #{n}" for n in targets if n in by_atomic]
                current_idx = 0
                cur = st.session_state.get("debug_target_override_atomic")
                if cur and cur in targets:
                    for i, n in enumerate(targets, start=1):
                        if n == cur:
                            current_idx = i
                            break

                choice = st.selectbox("Force target element (for testing)", options, index=current_idx)
                if choice == "— none —":
                    st.session_state.debug_target_override_atomic = None
                else:
                    st.session_state.debug_target_override_atomic = int(choice.split("#")[-1])

                can_undo = (st.session_state.status == "playing") and (len(st.session_state.history) > 0)
                if st.button("↩️ Undo last valid guess", disabled=not can_undo, use_container_width=True):
                    undo_snapshot()
                    st.session_state.ui_message = "Undid last guess."
                    st.rerun()

                with st.expander("📈 Last 3 runs per target", expanded=True):
                    _render_test_metrics(by_atomic, targets)

                with st.expander("🔎 Debug analytics", expanded=False):
                    # Show analytics later in main area too; here is just convenience.
                    pass

    # Choose effective secret for this run
    if st.session_state.debug_enabled and st.session_state.debug_target_override_atomic:
        secret = by_atomic[st.session_state.debug_target_override_atomic]
        st.warning(f"🧪 Debug: Target overridden to **{secret.name} ({secret.symbol})**")
    else:
        secret = st.session_state.secret

    guessed: Set[int] = set(st.session_state.guesses)

    # UI message (non-destructive)
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

    # Debug analytics in main area (if enabled)
    if st.session_state.debug_enabled:
        with st.expander("🛠 Debug analytics", expanded=False):
            show_debug_panel(secret, elements, st.session_state.revealed)

    # Prepare tiles
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

    board_key = f"tiles_board_{st.session_state.board_nonce}"

    # Render periodic table component.
    # Some users have a periodic_table() signature without invalidAtomic; we support both.
    try:
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
    except TypeError:
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

            if st.session_state.debug_enabled:
                push_snapshot()

            st.session_state.last_valid_guess_atomic = atomic
            st.session_state.last_guess_atomic = atomic

            # Guess feedback
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

            # Reveal next clue using max-pool logic (skips useless clues automatically)
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

    # Guesses list
    if st.session_state.guesses:
        st.subheader("Guesses")
        labels = [f"{by_atomic[a].name} ({by_atomic[a].symbol})" for a in st.session_state.guesses]
        st.write(" • ".join(labels))

    # Share link (stable URL on Streamlit Cloud)
    st.subheader("🔗 Share this game")
    copy_link_button()
    st.caption("Tip: You can also copy the URL from your browser’s address bar.")

    # Record stats exactly once on game end
    if st.session_state.status in ("won", "lost") and not st.session_state.stats_recorded:
        stats = load_stats()
        if st.session_state.status == "won":
            stats = record_win(stats, len(st.session_state.guesses))
        else:
            stats = record_loss(stats)
        save_stats(stats)

        # If debug override is active, record test harness metrics too
        if st.session_state.debug_enabled and st.session_state.debug_target_override_atomic:
            _record_test_run(st.session_state.debug_target_override_atomic)

        st.session_state.stats_recorded = True

    # End-game summary + share results
    if st.session_state.status in ("won", "lost"):
        if st.session_state.status == "won":
            st.success(f"🎉 Correct! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")
        else:
            st.error(f"😅 Out of guesses! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")
            st.caption("⬛ The answer has been revealed on the periodic table.")

        render_endgame_summary(secret, st.session_state.revealed, st.session_state.status)

        st.subheader("📋 Share results")
        share_text = build_share_text(secret)
        copy_to_clipboard_button(share_text, label="📋 Copy results")
        with st.expander("Preview share text", expanded=False):
            st.code(share_text)
    else:
        st.caption("Finish the game to unlock the Element Summary + Share Results.")

    st.markdown("---")
    st.caption(f"🧪 Element Guess • {APP_VERSION} • GitHub: {GITHUB_URL}")


if __name__ == "__main__":
    main()
