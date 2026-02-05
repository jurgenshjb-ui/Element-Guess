from __future__ import annotations

import datetime as dt
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st
import streamlit.components.v1 as components

from element_tiles import periodic_table

# =========================================================
# Public app metadata
# =========================================================
APP_VERSION = "v1.1.0"
GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO"

# Streamlit Cloud: put in Secrets (TOML):
# STREAMLIT_ENV = "prod"
IS_PRODUCTION = (
    os.environ.get("STREAMLIT_ENV") == "prod"
    or st.secrets.get("STREAMLIT_ENV", "dev") == "prod"
)


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

    # new/optional fields from JSON
    is_noble_gas: bool
    boiling_point_K: Optional[float]
    bohr_model_image: str


# =========================================================
# Chemistry corrections (in-code, so even if JSON is wrong,
# the game remains correct)
# =========================================================
NOBLE_GAS_NAMES = {"Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon", "Oganesson"}
CATEGORY_OVERRIDES = {"Hydrogen": "nonmetal", **{n: "nonmetal" for n in NOBLE_GAS_NAMES}}


def normalize_state(phase: str) -> str:
    t = (phase or "").lower().strip()
    return t if t in ("solid", "liquid", "gas") else "unknown"


def normalize_category3(raw_category: str, name: str) -> str:
    # hard override for known mislabels
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
    # If dataset lacks flags but group exists, group 18 is noble gases
    return group == 18


# =========================================================
# Radioactivity rule (game rule)
# Tc (43) and Pm (61) have no stable isotopes
# All Z >= 84 have no stable isotopes
# =========================================================
def is_radioactive(e: Element) -> bool:
    return (e.atomic_number in (43, 61)) or (e.atomic_number >= 84)


def origin_natural_vs_synthetic(e: Element) -> str:
    # only meaningful after Radioactive: Yes is revealed
    if e.atomic_number >= 93:
        return "synthetic"
    if e.atomic_number in (43, 61):
        return "synthetic"
    return "natural"


def is_lanthanoid_or_actinoid(e: Element) -> bool:
    return (57 <= e.atomic_number <= 71) or (89 <= e.atomic_number <= 103)


def group_family(e: Element) -> str:
    # Updated: includes lanthanoid/actinoid grouping
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
        name = r.get("name", "")
        symbol = r.get("symbol", "")
        number = int(r.get("number"))

        group = r.get("group")
        period = r.get("period")

        # state from "phase" in your JSON
        state = normalize_state(r.get("phase"))

        # category override + normalization
        category3 = normalize_category3(r.get("category"), name)

        # is_noble_gas: prefer explicit field if present, otherwise compute
        is_ng = bool(r.get("is_noble_gas")) if ("is_noble_gas" in r) else compute_is_noble_gas(name, group)

        # boiling point K: prefer explicit boiling_point_K; otherwise try common datasets field "boil" (Kelvin)
        bp = r.get("boiling_point_K", None)
        if bp is None and isinstance(r.get("boil"), (int, float)):
            bp = float(r["boil"])
        elif isinstance(bp, (int, float)):
            bp = float(bp)
        else:
            bp = None

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
    "radioactive": "Radioactive",
    "origin": "Origin (natural / synthetic)",
    "metal_group": "Metal group",
}

# Reveal order preference (you can tweak later)
# Phase/state is before noble_gas; noble_gas only eligible if state == gas
CLUE_ORDER = [
    "category3",
    "group_family",
    "block",
    "period",
    "band",
    "state",
    "noble_gas",
    "boil_split",
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
    # boil_split is handled specially in matches()
    return "unknown"


def matches(e: Element, revealed: Dict[str, str]) -> bool:
    for k, v in revealed.items():
        if k == "boil_split":
            # v format: "le|240" or "gt|240"
            if e.boiling_point_K is None:
                return False
            op, xs = v.split("|", 1)
            x = float(xs)
            if op == "le" and not (e.boiling_point_K <= x):
                return False
            if op == "gt" and not (e.boiling_point_K > x):
                return False
            continue

        if prop(e, k) != v:
            return False
    return True


def candidates(elements: List[Element], revealed: Dict[str, str], guessed: Set[int]) -> List[Element]:
    return [e for e in elements if e.atomic_number not in guessed and matches(e, revealed)]


def allowed_clue_keys(revealed: Dict[str, str], attempt: int) -> List[str]:
    keys = [k for k in CLUE_ORDER if k not in revealed]

    # Gate radioactive until after the 3rd guess
    if attempt < 3 and "radioactive" in keys:
        keys.remove("radioactive")

    # Gate origin until radioactive is revealed AND it's Yes
    if "origin" in keys and revealed.get("radioactive") != "Yes":
        keys.remove("origin")

    # Gate metal_group until category is metal
    if "metal_group" in keys and revealed.get("category3") != "metal":
        keys.remove("metal_group")

    # Phase/state must come before noble gas, and noble gas only eligible if state == gas
    if "noble_gas" in keys and revealed.get("state") != "gas":
        keys.remove("noble_gas")

    return keys


def debug_unrevealed_keys(revealed: Dict[str, str]) -> List[str]:
    # Debug ignores gameplay gating
    return [k for k in CLUE_ORDER if k not in revealed]


# =========================================================
# Dynamic boiling split (≤/> X) where X is rounded to nearest 10
# and based on current remaining candidates
# =========================================================
def choose_boiling_split_threshold(current_candidates: List[Element]) -> Optional[float]:
    vals = sorted([e.boiling_point_K for e in current_candidates if e.boiling_point_K is not None])
    if len(vals) < 6:
        return None
    mid = vals[len(vals) // 2]
    x = round(mid / 10.0) * 10.0
    return float(x)


def eval_secret_side_count_for_boiling(
    secret: Element, current_candidates: List[Element], x: float
) -> Optional[Tuple[str, int]]:
    if secret.boiling_point_K is None:
        return None
    side = "le" if secret.boiling_point_K <= x else "gt"
    cnt = 0
    for e in current_candidates:
        if e.boiling_point_K is None:
            continue
        if side == "le" and e.boiling_point_K <= x:
            cnt += 1
        elif side == "gt" and e.boiling_point_K > x:
            cnt += 1
    return side, cnt


def choose_next_clue(
    secret: Element,
    elements: List[Element],
    revealed: Dict[str, str],
    attempt: int,
) -> Optional[Tuple[str, str, int]]:
    """
    Choose next clue with the WIDEST possible remaining candidate pool
    BUT ignore any clue that doesn't narrow at all.
    """
    remaining_keys = allowed_clue_keys(revealed, attempt)
    if not remaining_keys:
        return None

    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    if current_n <= 1:
        return None

    best_key: Optional[str] = None
    best_value: Optional[str] = None
    best_count = -1

    # Prefer only strictly narrowing clues (0 < cnt < current_n)
    for k in remaining_keys:
        if k == "boil_split":
            x = choose_boiling_split_threshold(current_candidates)
            if x is None:
                continue
            res = eval_secret_side_count_for_boiling(secret, current_candidates, x)
            if res is None:
                continue
            side, cnt = res
            if 0 < cnt < current_n and cnt > best_count:
                best_key = "boil_split"
                best_value = f"{side}|{int(x)}"
                best_count = cnt
            continue

        v = prop(secret, k)
        cnt = sum(1 for e in current_candidates if prop(e, k) == v)
        if 0 < cnt < current_n and cnt > best_count:
            best_key = k
            best_value = v
            best_count = cnt

    # Fallback if nothing narrows (rare): pick largest group anyway
    if best_key is None:
        for k in remaining_keys:
            if k == "boil_split":
                x = choose_boiling_split_threshold(current_candidates)
                if x is None:
                    continue
                res = eval_secret_side_count_for_boiling(secret, current_candidates, x)
                if res is None:
                    continue
                side, cnt = res
                if cnt > best_count:
                    best_key = "boil_split"
                    best_value = f"{side}|{int(x)}"
                    best_count = cnt
                continue

            v = prop(secret, k)
            cnt = sum(1 for e in current_candidates if prop(e, k) == v)
            if cnt > best_count:
                best_key = k
                best_value = v
                best_count = cnt

    if best_key is None or best_value is None:
        return None
    return best_key, best_value, best_count


def reveal_next_clue(secret: Element, elements: List[Element], revealed: Dict[str, str], attempt: int) -> None:
    nxt = choose_next_clue(secret, elements, revealed, attempt)
    if nxt:
        k, v, _ = nxt
        if k not in revealed:
            revealed[k] = v
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
- **Noble gases (Group 18)**: very stable, mostly unreactive nonmetals.
- **Lanthanoid or actinoid**: elements in **lanthanoids (57–71)** or **actinoids (89–103)**.

**Electron Blocks**
- **s-block**: Groups 1–2 (plus H, He).
- **p-block**: Groups 13–18.
- **d-block**: Transition metals (Groups 3–12).
- **f-block**: Lanthanoids & actinoids (shown separately).

**Radioactive (game rule)**
- **Radioactive: Yes** if the element has **no stable isotopes**.
- In this game: **Tc (43)** and **Pm (61)** are radioactive, and **all elements with Z ≥ 84** are radioactive.

**Boiling point split (K)**
- The game may reveal **Boiling point: ≤ X K** or **> X K**, where **X** is chosen to split remaining candidates roughly in half.
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
        if k == "boil_split":
            op, xs = v.split("|", 1)
            sign = "≤" if op == "le" else ">"
            ok = matches(e, {k: v})
            lines.append(f"{'✅' if ok else '❌'} {CLUES[k]}: {sign} {xs} K")
        else:
            ok = (prop(e, k) == v)
            lines.append(f"{'✅' if ok else '❌'} {CLUES.get(k, k)}: {v}")
    return "\n".join(lines)


# =========================================================
# Daily / Infinite selection
# =========================================================
def pick_daily(elements: List[Element], date: Optional[dt.date] = None) -> Element:
    date = date or dt.date.today()
    start = dt.date(2026, 1, 1)
    idx = abs((date - start).days) % len(elements)
    return elements[idx]


def pick_infinite(elements: List[Element], seed: int) -> Element:
    return random.Random(seed).choice(elements)


# =========================================================
# Special clue (Higher / Lower) with toggle
# =========================================================
def compute_hi_lo_special_clue(
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

    remaining = candidates(elements, revealed, guessed)
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
    for e in current_candidates:
        if clue_key == "boil_split":
            # show boiling distribution as raw numeric bands for debug (not used as gameplay buckets)
            if e.boiling_point_K is None:
                v = "unknown"
            else:
                v = f"{int(round(e.boiling_point_K / 10.0) * 10)} K (rounded)"
        else:
            v = prop(e, clue_key)
        counts[v] = counts.get(v, 0) + 1

    total = max(len(current_candidates), 1)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))

    out = []
    for v, c in ordered:
        pct = (c / total) * 100.0
        out.append({"value": v, "count": c, "percent": f"{pct:.1f}%"})
    return out


def show_debug_panel(secret: Element, elements: List[Element], revealed: Dict[str, str], attempt: int):
    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    st.write(f"**Current candidates (matching revealed):** {current_n}")

    # 1) Secret-value impact + delta reduction (debug ignores gating)
    rows = []
    for k in debug_unrevealed_keys(revealed):
        if k == "boil_split":
            x = choose_boiling_split_threshold(current_candidates)
            if x is None:
                continue
            res = eval_secret_side_count_for_boiling(secret, current_candidates, x)
            if res is None:
                continue
            side, cnt = res
            secret_value = f"{'≤' if side=='le' else '>'} {int(x)} K"
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
# Table positioning + tile building
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
    """
    Tile status values passed to TSX:
      none, bad, close, correct, hint, lost
    """
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
            # animation flags consumed by TSX
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
# Share results helpers
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


# =========================================================
# State helpers
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

    st.session_state.ui_message = None
    st.session_state.history = []

    st.session_state.prev_hint_set = set()
    st.session_state.invalid_atomic = None
    st.session_state.last_guess_atomic = None
    st.session_state.win_anim_pending = False

    if game_mode == "Daily":
        st.session_state.secret = pick_daily(elements)
        st.session_state.seed = None
    else:
        seed = random.randrange(1_000_000_000)
        st.session_state.seed = seed
        st.session_state.secret = pick_infinite(elements, seed)


def ensure_state(elements: List[Element]):
    if "game_mode" not in st.session_state:
        st.session_state.game_mode = "Daily"
        st.session_state.difficulty = "Normal"
        st.session_state.enable_special_clue = True
        st.session_state.debug_secret_atomic = None
        start_new_game(elements, st.session_state.game_mode)

    st.session_state.setdefault("enable_special_clue", True)
    st.session_state.setdefault("debug_secret_atomic", None)
    st.session_state.setdefault("ui_message", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("revealed_order", [])
    st.session_state.setdefault("prev_hint_set", set())
    st.session_state.setdefault("invalid_atomic", None)
    st.session_state.setdefault("last_guess_atomic", None)
    st.session_state.setdefault("win_anim_pending", False)


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
# End-game summary (only show properties that were revealed
# or partially revealed by clues)
# =========================================================
def format_boiling_point(bp: Optional[float]) -> str:
    if bp is None:
        return "unknown"
    # show rounded to nearest 1K for readability
    return f"{bp:.0f} K"


def revealed_property_rows(secret: Element, revealed: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Return rows for the end-game summary table.
    Include only properties that were revealed or partially revealed.
    """
    rows: List[Dict[str, str]] = []

    # Map each clue key to an "actual value" display + attribution
    for k in st.session_state.revealed_order:
        if k not in revealed:
            continue

        if k == "boil_split":
            op, xs = revealed[k].split("|", 1)
            sign = "≤" if op == "le" else ">"
            rows.append(
                {
                    "Property": "Boiling point (K)",
                    "Value": f"{format_boiling_point(secret.boiling_point_K)}",
                    "Clue": f"{sign} {xs} K",
                    "Revealed by clue": "✅ (partial)",
                }
            )
            continue

        # standard clue properties (full reveal)
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
    st.subheader(f"🧾 Element Summary — {secret.name} ({secret.symbol})")

    # image + key facts
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        if secret.bohr_model_image:
            try:
                st.image(secret.bohr_model_image, caption=f"Bohr model — {secret.name} ({secret.symbol})", use_container_width=True)
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
**Boiling point:** {format_boiling_point(secret.boiling_point_K)}  
**Radioactive (game rule):** {"Yes" if is_radioactive(secret) else "No"}  
"""
        )

    rows = revealed_property_rows(secret, revealed)
    if rows:
        st.markdown("**What the clues revealed (and/or implied):**")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No clues were revealed in this round (rare).")

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


# =========================================================
# Main app
# =========================================================
def main():
    st.set_page_config(page_title="Element Guess", page_icon="🧪", layout="wide")
    inject_mobile_css()

    st.title("🧪 Element Guess")
    st.caption("Click an element tile to guess. Colors show your progress.")

    # How to play expander
    with st.expander("❓ How to play", expanded=False):
        st.markdown(
            """
**Goal:** Guess the hidden element in **7 guesses or fewer**.

**How it works:**
- Click an element on the periodic table to guess.
- After each guess, the game reveals **shared properties** of the target element.
- Use those clues to narrow down your next guess.

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
"""
        )

    elements = load_elements()
    by_atomic = {e.atomic_number: e for e in elements}

    ensure_state(elements)

    can_change_settings = (st.session_state.status in ("won", "lost")) or (len(st.session_state.guesses) == 0)

    debug_mode = False

    with st.sidebar:
        st.subheader("Game")

        mode_choice = st.radio(
            "Mode",
            ["Daily", "Infinite"],
            index=["Daily", "Infinite"].index(st.session_state.game_mode),
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

        st.subheader("Hints")
        st.session_state.enable_special_clue = st.toggle(
            "Enable special clue (Higher/Lower)",
            value=bool(st.session_state.enable_special_clue),
        )

        if not IS_PRODUCTION:
            debug_mode = st.toggle("Debug mode", value=False)

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

        if st.button("🔄 Restart"):
            start_new_game(elements, st.session_state.game_mode)
            st.rerun()

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
    special_hi_lo = compute_hi_lo_special_clue(
        enabled=bool(st.session_state.enable_special_clue),
        elements=elements,
        revealed=st.session_state.revealed,
        guessed=guessed,
        guesses_left=guesses_left,
        secret=secret,
        last_valid_guess_atomic=st.session_state.last_valid_guess_atomic,
    )

    # Debug panel
    if debug_mode:
        with st.expander("🛠 Debug", expanded=True):
            show_debug_panel(secret, elements, st.session_state.revealed, st.session_state.attempt)

    # One-shot UI triggers
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

    click = periodic_table(
        tiles=tiles,
        legend={
            "none": "#E5E7EB",
            "bad": "#EF4444",
            "close": "#F59E0B",
            "correct": "#16A34A",
            "hint": "#3B82F6",
            "lost": "#111827",  # charcoal reveal on loss
        },
        disabled=(st.session_state.status != "playing"),
        invalidAtomic=invalid_atomic_to_send,
        key="tiles_board",
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
            st.session_state.last_guess_atomic = atomic  # highlight once

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

            reveal_next_clue(secret, elements, st.session_state.revealed, st.session_state.attempt)

            if st.session_state.attempt >= st.session_state.max_guesses:
                st.session_state.status = "lost"

            st.rerun()

    # Revealed clues — numbered, chronological
    if st.session_state.revealed or special_hi_lo:
        st.subheader("Revealed clues")
        i = 1
        for k in st.session_state.revealed_order:
            if k not in st.session_state.revealed:
                continue

            v = st.session_state.revealed[k]
            if k == "boil_split":
                op, xs = v.split("|", 1)
                sign = "≤" if op == "le" else ">"
                st.write(f"{i}. **{CLUES[k]}:** {sign} {xs} K")
            else:
                st.write(f"{i}. **{CLUES[k]}:** {v}")
            i += 1

        if special_hi_lo:
            st.write(f"{i}. **Special clue:** {special_hi_lo}")

    if st.session_state.guesses:
        st.subheader("Guesses")
        labels = [f"{by_atomic[a].name} ({by_atomic[a].symbol})" for a in st.session_state.guesses]
        st.write(" • ".join(labels))

    # Share link section
    st.subheader("🔗 Share this game")
    copy_link_button()
    st.caption("Tip: You can also copy the URL from your browser’s address bar.")

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

    # Footer
    st.markdown("---")
    st.caption(f"🧪 Element Guess • {APP_VERSION} • GitHub: {GITHUB_URL}")


if __name__ == "__main__":
    main()
