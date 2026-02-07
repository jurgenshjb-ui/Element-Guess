from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

# Your custom periodic-table component
from element_tiles import periodic_table

# =========================================================
# App metadata (edit to match your repo)
# =========================================================
APP_VERSION = "v1.4.1"
GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO"
WIKI_BASE = "https://en.wikipedia.org/wiki/"
RSC_SEARCH = "https://www.rsc.org/periodic-table/element/"
WEBELEMENTS_BASE = "https://www.webelements.com/"

# =========================================================
# Production / Debug gating
# Streamlit Cloud -> Secrets:
# STREAMLIT_ENV = "prod"
# DEBUG = "false"
# =========================================================
IS_PRODUCTION = (
    os.environ.get("STREAMLIT_ENV") == "prod"
    or st.secrets.get("STREAMLIT_ENV", "dev") == "prod"
)
DEBUG_ALLOWED = str(st.secrets.get("DEBUG", "false")).lower() == "true"

# Debug is always available when running locally/dev.
# On Streamlit Cloud, keep it hidden by default (STREAMLIT_ENV="prod").
# To enable debug on Cloud temporarily, set DEBUG="true" in Secrets.
SHOW_DEBUG_UI = (not IS_PRODUCTION) or DEBUG_ALLOWED

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

        bp = r.get("boiling_point_K", None)
        if bp is None and isinstance(r.get("boil"), (int, float)):
            bp = float(r["boil"])
        elif isinstance(bp, (int, float)):
            bp = float(bp)
        else:
            bp = None

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
    "starter": "Starter info (doesn't filter guesses)",
    "category3": "Category (metal / nonmetal / metalloid)",
    "group_family": "Major group family",
    "block": "Electron block (s / p / d / f)",
    "band": "Atomic number range",
    "state": "State at room temp",
    "noble_gas": "Noble gas",
    "boil_split": "Boiling point (K)",
    "melt_split": "Melting point (K)",
    "period": "Period",
    "radioactive": "Radioactive",
    "origin": "Origin (natural / synthetic)",
    "metal_group": "Metal group",
}

CLUE_ORDER = [
    "category3",
    "group_family",
    "block",
    "band",
    "state",
    "noble_gas",
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
    return e.boiling_point_K if kind == "boil" else e.melting_point_K

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

def _choose_temp_split_threshold(current_candidates: List[Element], kind: str) -> Optional[float]:
    known_vals = sorted([_temp_value_K(e, kind) for e in current_candidates if _temp_value_K(e, kind) is not None])
    total = len(current_candidates)
    known = len(known_vals)
    if total == 0 or known < 6:
        return None
    if (known / total) < 0.70:
        return None
    median = known_vals[len(known_vals) // 2]
    x = round(median / 100.0) * 100.0
    return float(x)

def _count_candidates_for_secret_temp_value(secret: Element, current_candidates: List[Element], clue_key: str, x: float) -> Optional[Tuple[str, int]]:
    kind = "boil" if clue_key == "boil_split" else "melt"
    val_s = _temp_value_K(secret, kind)
    if val_s is None:
        return None
    bucket = "≤X" if val_s <= x else ">X"
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

def _period_is_eligible(attempt: int, current_n: int) -> bool:
    return (attempt >= 4) and (current_n > 18)

def allowed_clue_keys(revealed: Dict[str, str], attempt: int, current_n: int) -> List[str]:
    keys = [k for k in CLUE_ORDER if k not in revealed]
    if attempt < 3 and "radioactive" in keys:
        keys.remove("radioactive")
    if attempt < 3:
        for k in ("boil_split", "melt_split"):
            if k in keys:
                keys.remove(k)
    if "boil_split" in revealed and "melt_split" in keys:
        keys.remove("melt_split")
    if "melt_split" in revealed and "boil_split" in keys:
        keys.remove("boil_split")
    if "origin" in keys and revealed.get("radioactive") != "Yes":
        keys.remove("origin")
    if "metal_group" in keys and revealed.get("category3") != "metal":
        keys.remove("metal_group")
    if "noble_gas" in keys and revealed.get("state") != "gas":
        keys.remove("noble_gas")
    if "period" in keys and (not _period_is_eligible(attempt, current_n)):
        keys.remove("period")
    return keys

def _deterministic_tiebreak(game_mode: str, seed: Optional[int]) -> int:
    if game_mode == "Daily":
        return dt.date.today().toordinal() % 2
    return (seed or 0) % 2

def choose_next_clue_max_pool(secret: Element, elements: List[Element], revealed: Dict[str, str], attempt: int, game_mode: str, seed: Optional[int]) -> Optional[Tuple[str, str, int]]:
    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    if current_n <= 1:
        return None
    remaining_keys = allowed_clue_keys(revealed, attempt, current_n)
    if not remaining_keys:
        return None

    boil_x = _choose_temp_split_threshold(current_candidates, "boil") if "boil_split" in remaining_keys else None
    melt_x = _choose_temp_split_threshold(current_candidates, "melt") if "melt_split" in remaining_keys else None

    best = None  # (key, value, cnt, tie_rank, ord_idx)
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
                elif ord_idx == best_ord and tie_rank < best_tie:
                    best = cand

    if best is None:
        return None
    k, v, cnt, _, _ = best
    return k, v, cnt

def reveal_next_clue(secret: Element, elements: List[Element], attempt: int) -> None:
    nxt = choose_next_clue_max_pool(secret, elements, st.session_state.revealed, attempt, st.session_state.game_mode, st.session_state.seed)
    if not nxt:
        return
    k, v, _ = nxt
    st.session_state.revealed[k] = v
    st.session_state.revealed_order.append(k)

def definitions_panel():
    with st.expander("📘 Definitions (Categories, Groups, Blocks, Radioactive)", expanded=False):
        st.markdown(
            """
**Electron Blocks on the table**
- **s-block**: Groups 1–2 (plus H, He)
- **p-block**: Groups 13–18
- **d-block**: Transition metals (Groups 3–12)
- **f-block**: Lanthanoids & actinoids

**Radioactive (game rule)**
- Tc (43) and Pm (61) are radioactive
- All elements with Z ≥ 84 are radioactive

**Temperature split clues**
- Reveals **Boiling** or **Melting**: **≤ X K** or **> X K**
- Only one temperature clue appears per game
- X is rounded to the nearest **100 K**
"""
        )

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

def how_to_play():
    with st.expander("❓ How to play", expanded=False):
        st.markdown(
            """
**Goal:** Guess the hidden element in **8 guesses or fewer**.

**Tile colors:**
- 🟥 **Red** — fails revealed clues
- 🟧 **Amber** — matches clues, but not the answer
- 🟩 **Green** — correct
- 🟦 **Blue** — possible candidates (Easy mode)
- ⬛ **Charcoal** — revealed answer (loss)

**Blocks overlay:** The table outlines the **s/p/d/f** blocks (legend included).
"""
        )

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

def make_tiles(elements: List[Element], revealed: Dict[str, str], guessed: Set[int], secret_atomic: int, difficulty: str, last_guess_atomic: Optional[int], win_anim_pending: bool, status: str) -> List[dict]:
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

def start_new_game(elements: List[Element], game_mode: str):
    st.session_state.status = "playing"
    st.session_state.guesses = []
    st.session_state.guess_feedback = []
    st.session_state.revealed = {}
    st.session_state.revealed_order = []
    st.session_state.attempt = 0
    st.session_state.max_guesses = 8
    st.session_state.last_click_nonce = None
    st.session_state.last_valid_guess_atomic = None
    st.session_state.last_guess_atomic = None
    st.session_state.invalid_atomic = None
    st.session_state.win_anim_pending = False
    st.session_state.ui_message = None
    st.session_state.prev_hint_set = set()

    st.session_state.board_nonce = int(st.session_state.get("board_nonce", 0)) + 1

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
        st.session_state.debug_enabled = False
        st.session_state.board_nonce = 0
        st.session_state.prev_hint_set = set()
        start_new_game(elements, st.session_state.game_mode)

    st.session_state.setdefault("ui_message", None)
    st.session_state.setdefault("prev_hint_set", set())
    st.session_state.setdefault("invalid_atomic", None)
    st.session_state.setdefault("last_guess_atomic", None)
    st.session_state.setdefault("win_anim_pending", False)
    st.session_state.setdefault("board_nonce", 0)
    st.session_state.setdefault("revealed_order", [])

def starter_info(secret: Element) -> Optional[str]:
    # Informational only. Prefer discovered_by, else named_by, else source.
    if secret.discovered_by:
        return f"Discovered by: {secret.discovered_by}"
    if secret.named_by:
        return f"Named by: {secret.named_by}"
    if secret.source:
        return f"Source: {secret.source}"
    return None

def main():
    st.set_page_config(page_title="Element Guess", page_icon="🧪", layout="wide")
    st.title("🧪 Element Guess")
    st.caption("Click an element tile to guess. Colors show your progress.")

    how_to_play()

    elements = load_elements()
    by_atomic = {e.atomic_number: e for e in elements}
    ensure_state(elements)

    # Settings change guard
    can_change_settings = (st.session_state.status in ("won", "lost")) or (len(st.session_state.guesses) == 0)

    # Sidebar
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


        if SHOW_DEBUG_UI:
            st.divider()
            st.session_state.debug_enabled = st.toggle("🛠 Debug mode", value=bool(st.session_state.debug_enabled))

            if st.session_state.debug_enabled:
                st.subheader("🧰 Debug tools")
                st.session_state.max_guesses = int(
                    st.number_input(
                        "Max guesses (debug)",
                        min_value=1,
                        max_value=20,
                        value=int(st.session_state.max_guesses),
                        step=1,
                        help="Adjust how many guesses you get. This is for testing only.",
                    )
                )
                if st.button("🔄 Restart (keep debug settings)", use_container_width=True):
                    start_new_game(elements, st.session_state.game_mode)
                    st.rerun()

                with st.expander("🧮 Candidate count", expanded=False):
                    cand = [e for e in elements if matches(e, st.session_state.revealed)]
                    st.write(f"Candidates matching revealed clues: **{len(cand)}**")


    # Header status
    st.write(f"🧩 **Mode:** {st.session_state.game_mode}   |   🎚️ **Difficulty:** {st.session_state.difficulty}")
    st.write(f"🎯 **Guesses used:** {st.session_state.attempt}/{st.session_state.max_guesses}")

    # Choose secret
    secret = st.session_state.secret
    guessed: Set[int] = set(st.session_state.guesses)

    # Starter clue (Easy+Normal only)
    if st.session_state.difficulty in ("Easy", "Normal"):
        info = starter_info(secret)
        if info:
            st.info(f"🟦 **{CLUES['starter']}:** {info}")

    # Revealed clues ABOVE the table
    if st.session_state.revealed:
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

    # Build tiles + render component
    tiles = make_tiles(
        elements=elements,
        revealed=st.session_state.revealed,
        guessed=guessed,
        secret_atomic=secret.atomic_number,
        difficulty=st.session_state.difficulty,
        last_guess_atomic=st.session_state.last_guess_atomic,
        win_anim_pending=bool(st.session_state.win_anim_pending),
        status=st.session_state.status,
    )

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
            "s": "#60A5FA",
            "p": "#FBBF24",
            "d": "#34D399",
            "f": "#A78BFA",
        },
        disabled=(st.session_state.status != "playing"),
        key=board_key,
    )

    # Handle click
    if click and st.session_state.status == "playing":
        atomic = click.get("atomic")
        nonce = click.get("nonce")
        if nonce is not None and nonce == st.session_state.last_click_nonce:
            pass
        else:
            st.session_state.last_click_nonce = nonce
            if not isinstance(atomic, int) or atomic not in by_atomic:
                st.rerun()
            e = by_atomic[atomic]

            # Normal + Hard: invalid guess doesn't consume a guess; shake + message
            if st.session_state.revealed and (not matches(e, st.session_state.revealed)):
                if st.session_state.difficulty in ("Normal", "Hard"):
                    st.session_state.ui_message = "Not a valid choice — try a different element."
                    st.rerun()

            if atomic in guessed:
                st.session_state.ui_message = f"You already guessed {e.name} ({e.symbol})."
                st.rerun()

            st.session_state.last_valid_guess_atomic = atomic
            st.session_state.last_guess_atomic = atomic

            st.session_state.guesses.append(atomic)
            st.session_state.attempt += 1

            if atomic == secret.atomic_number:
                st.session_state.status = "won"
                st.session_state.win_anim_pending = True
                st.rerun()

            # Reveal next clue
            reveal_next_clue(secret, elements, st.session_state.attempt)

            if st.session_state.attempt >= st.session_state.max_guesses:
                st.session_state.status = "lost"

            st.rerun()

    # Move definitions UNDER the table
    definitions_panel()

    st.markdown("---")
    st.caption(f"🧪 Element Guess • {APP_VERSION} • GitHub: {GITHUB_URL}")

if __name__ == "__main__":
    main()
