from __future__ import annotations

import datetime as dt
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import streamlit as st
import streamlit.components.v1 as components
from element_tiles import periodic_table


# =========================================================
# Public app metadata (edit these)
# =========================================================
APP_VERSION = "v1.0.0"
GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO"

# Hide debug tools on deployed app by setting STREAMLIT_ENV=prod in Streamlit Cloud
IS_PRODUCTION = os.environ.get("STREAMLIT_ENV") == "prod" or st.secrets.get("STREAMLIT_ENV", "dev") == "prod"


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
    state: str
    radioactive_raw: bool
    summary: str
    discovered_by: str
    named_by: str
    source: str


# =========================================================
# Normalization / property helpers
# =========================================================
def normalize_category3(t: str) -> str:
    t = (t or "").lower()
    if "metalloid" in t:
        return "metalloid"
    if "nonmetal" in t:
        return "nonmetal"
    return "metal"


def normalize_state(t: str) -> str:
    t = (t or "").lower().strip()
    return t if t in ("solid", "liquid", "gas") else "unknown"


def is_lanthanoid_or_actinoid(e: Element) -> bool:
    return (57 <= e.atomic_number <= 71) or (89 <= e.atomic_number <= 103)


def group_family(e: Element) -> str:
    """
    Updated: includes "lanthanoid or actinoid" grouping to reduce "other/none".
    """
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
# Radioactivity rule (FIXED)
#   - Tc (43) and Pm (61) have no stable isotopes
#   - All elements with Z >= 84 have no stable isotopes
# =========================================================
def is_radioactive(e: Element) -> bool:
    return (e.atomic_number in (43, 61)) or (e.atomic_number >= 84)


def origin_natural_vs_synthetic(e: Element) -> str:
    """
    Only used after Radioactive: Yes is revealed.
    Gameplay-friendly classification:
      - Synthetic: elements >= 93, plus Technetium (43), Promethium (61)
      - Natural: everything else
    """
    if e.atomic_number >= 93:
        return "synthetic"
    if e.atomic_number in (43, 61):
        return "synthetic"
    return "natural"


def metal_group(e: Element) -> str:
    """
    Only meaningful when Category is 'metal'.
    Buckets:
      - alkali metals
      - alkaline earth metals
      - transition metals
      - inner transition metals
      - post-transition metals
    """
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


# =========================================================
# Load elements
# =========================================================
@st.cache_data
def load_elements(path: str = "PeriodicTableJSON.json") -> List[Element]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["elements"]

    out: List[Element] = []
    for r in raw:
        out.append(
            Element(
                name=r["name"],
                symbol=r["symbol"],
                atomic_number=r["number"],
                category3=normalize_category3(r.get("category")),
                group=r.get("group"),
                period=r.get("period"),
                state=normalize_state(r.get("phase")),
                radioactive_raw=bool(r.get("radioactive")),
                summary=(r.get("summary") or "").strip(),
                discovered_by=(r.get("discovered_by") or "").strip(),
                named_by=(r.get("named_by") or "").strip(),
                source=(r.get("source") or "").strip(),
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
    "state",
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
        return group_family(e)  # ✅ updated signature uses element
    if k == "block":
        return block_of(e)
    if k == "band":
        return atomic_band(e.atomic_number)
    if k == "radioactive":
        return "Yes" if is_radioactive(e) else "No"
    if k == "origin":
        return origin_natural_vs_synthetic(e)
    if k == "metal_group":
        return metal_group(e)
    return "unknown"


def matches(e: Element, revealed: Dict[str, str]) -> bool:
    return all(prop(e, k) == v for k, v in revealed.items())


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

    return keys


def debug_unrevealed_keys(revealed: Dict[str, str]) -> List[str]:
    # Debug ignores gameplay gating — show potential from the start
    return [k for k in CLUE_ORDER if k not in revealed]


def choose_next_clue(
    secret: Element,
    elements: List[Element],
    revealed: Dict[str, str],
    attempt: int,
) -> Optional[Tuple[str, str, int]]:
    """
    Choose next clue with the WIDEST possible remaining candidate pool,
    but IGNORE any clue that doesn't narrow at all (cnt == current_n).
    """
    remaining_keys = allowed_clue_keys(revealed, attempt)
    if not remaining_keys:
        return None

    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    if current_n <= 1:
        return None

    best_key = None
    best_count = -1

    # Pass 1: only clues that strictly narrow (0 < cnt < current_n)
    for k in remaining_keys:
        v = prop(secret, k)
        cnt = sum(1 for e in current_candidates if prop(e, k) == v)
        if 0 < cnt < current_n and cnt > best_count:
            best_key = k
            best_count = cnt

    # Pass 2: fallback if nothing narrows (rare)
    if best_key is None:
        for k in remaining_keys:
            v = prop(secret, k)
            cnt = sum(1 for e in current_candidates if prop(e, k) == v)
            if cnt > best_count:
                best_key = k
                best_count = cnt

    if best_key is None:
        return None
    return best_key, prop(secret, best_key), best_count


def reveal_next_clue(secret: Element, elements: List[Element], revealed: Dict[str, str], attempt: int) -> None:
    nxt = choose_next_clue(secret, elements, revealed, attempt)
    if nxt:
        k, v, _cnt = nxt
        if k not in revealed:
            revealed[k] = v
            st.session_state.revealed_order.append(k)


# =========================================================
# Definitions panel + CSV export
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
- **Noble gases (Group 18)**: very stable and unreactive.
- **Lanthanoid or actinoid**: elements in the **lanthanoids (57–71)** or **actinoids (89–103)** series.

**Electron Blocks**
- **s-block**: Groups 1–2 (plus H, He).
- **p-block**: Groups 13–18.
- **d-block**: Transition metals (Groups 3–12).
- **f-block**: Lanthanoids & actinoids (shown separately).

**Radioactive (game rule)**
- **Radioactive: Yes** if the element has **no stable isotopes**.
- In this game: **Tc (43)** and **Pm (61)** are radioactive, and **all elements with Z ≥ 84** are radioactive.
"""
        )


def build_definitions_grid_csv(elements: List[Element]) -> str:
    def fmt_list(es: List[Element]) -> str:
        return "; ".join([f"{e.name} ({e.symbol})" for e in sorted(es, key=lambda x: x.atomic_number)])

    rows: List[Tuple[str, str, List[Element]]] = []

    for cat in ("metal", "nonmetal", "metalloid"):
        rows.append(("Category", cat, [e for e in elements if e.category3 == cat]))

    fam_names = [
        "alkali metal (group 1)",
        "alkaline earth metal (group 2)",
        "halogen (group 17)",
        "noble gas (group 18)",
        "lanthanoid or actinoid",
        "other/none",
    ]
    for name in fam_names:
        rows.append(("Major group family", name, [e for e in elements if group_family(e) == name]))

    blocks = {"s-block": [], "p-block": [], "d-block": [], "f-block": [], "unknown": []}
    for e in elements:
        blocks[block_of(e)].append(e)
    for b in ("s-block", "p-block", "d-block", "f-block", "unknown"):
        rows.append(("Electron block", b, blocks[b]))

    rows.append(("Radioactive (game rule)", "Yes", [e for e in elements if is_radioactive(e)]))
    rows.append(("Radioactive (game rule)", "No", [e for e in elements if not is_radioactive(e)]))

    lines = ["Dimension,Bucket,Count,Elements"]
    for dim, bucket, es in rows:
        elements_str = fmt_list(es).replace('"', '""')
        lines.append(f'{dim},{bucket},{len(es)},"{elements_str}"')
    return "\n".join(lines)


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
# Debug mode helpers (impact + distributions + delta reduction)
# =========================================================
def _distribution_table(current_candidates: List[Element], clue_key: str) -> List[dict]:
    counts: Dict[str, int] = {}
    for e in current_candidates:
        v = prop(e, clue_key)
        counts[v] = counts.get(v, 0) + 1

    total = max(len(current_candidates), 1)
    ordered_keys = sorted(counts.keys(), key=lambda x: (-counts[x], x))

    out = []
    for k in ordered_keys:
        c = counts[k]
        pct = (c / total) * 100.0
        out.append({"value": k, "count": c, "percent": f"{pct:.1f}%"})
    return out


def show_debug_panel(secret: Element, elements: List[Element], revealed: Dict[str, str], attempt: int):
    current_candidates = [e for e in elements if matches(e, revealed)]
    current_n = len(current_candidates)
    st.write(f"**Current candidates (matching revealed):** {current_n}")

    # 1) Secret-value impact + delta reduction (debug ignores gating)
    rows = []
    for k in debug_unrevealed_keys(revealed):
        v = prop(secret, k)
        cnt = sum(1 for e in current_candidates if prop(e, k) == v)
        reduction = current_n - cnt
        reduction_pct = (reduction / max(current_n, 1)) * 100.0
        rows.append(
            {
                "clue_key": k,
                "clue": CLUES.get(k, k),
                "secret_value": v,
                "candidates_if_revealed": cnt,
                "reduction": reduction,
                "reduction_%": f"{reduction_pct:.1f}%",
            }
        )

    order_index = {k: i for i, k in enumerate(CLUE_ORDER)}
    rows.sort(key=lambda r: (-r["candidates_if_revealed"], order_index.get(r["clue_key"], 999)))

    if rows:
        st.markdown("**1) Unrevealed clue impact (secret value → candidates) + reduction (delta):**")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # 2) Distribution per clue (debug ignores gating — includes Radioactive from start)
    st.markdown("**2) Distribution per unrevealed clue (value → count, % of candidates):**")
    for k in debug_unrevealed_keys(revealed):
        with st.expander(f"{CLUES.get(k,k)} distribution", expanded=False):
            dist = _distribution_table(current_candidates, k)
            st.dataframe(dist, use_container_width=True, hide_index=True)


# =========================================================
# Animation + tile building
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
) -> List[dict]:
    main, lanth, actin = build_positions(elements)
    easy = (difficulty == "Easy")
    normal_locks = (difficulty == "Normal")

    # compute easy hint deltas (pulse only newly eligible)
    current_hint_set: Set[int] = set()
    if easy and revealed:
        for e in elements:
            if e.atomic_number not in guessed and matches(e, revealed):
                current_hint_set.add(e.atomic_number)

    prev_hint_set: Set[int] = set(st.session_state.prev_hint_set)
    new_hint_set = current_hint_set - prev_hint_set
    st.session_state.prev_hint_set = current_hint_set

    def status_of(e: Element) -> str:
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
# Share results + share link
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
    # Tries to copy the top-level URL; falls back to the iframe URL
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
# Debug undo (restore previous snapshots)
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

    # animation/ux state
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
# Main app
# =========================================================
def main():
    st.set_page_config(page_title="Element Guess", page_icon="🧪", layout="wide")
    inject_mobile_css()

    st.title("🧪 Element Guess")
    st.caption("Click an element tile to guess. Colors show your progress.")

    # How to play (small upgrade)
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

**Difficulty:**
- **Easy:** highlights valid candidates
- **Normal:** invalid guesses are locked
- **Hard:** invalid guesses are rejected (no explanation)
"""
        )

    elements = load_elements()
    by_atomic = {e.atomic_number: e for e in elements}

    ensure_state(elements)

    can_change_settings = (st.session_state.status in ("won", "lost")) or (len(st.session_state.guesses) == 0)

    # We only allow debug mode in dev, unless you explicitly set STREAMLIT_ENV!=prod
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

                choice = st.selectbox(
                    "Force target element",
                    options,
                    index=current_idx,
                    help="Overrides the secret element for analysis only (does not change real secret).",
                )

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

    # Date label for Daily
    if st.session_state.game_mode == "Daily":
        today = dt.date.today()
        st.info(f"📅 Today’s date: {today:%A, %d %B %Y}")

    definitions_panel()

    guessed: Set[int] = set(st.session_state.guesses)

    # Effective secret (debug override does not mutate real secret)
    if debug_mode and st.session_state.debug_secret_atomic:
        secret = by_atomic[st.session_state.debug_secret_atomic]
        st.warning(f"🧪 Debug mode: Target overridden to {secret.name} ({secret.symbol})")
        st.info("🔍 Debug override active — results and Daily integrity are NOT affected.")
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

    # One-shot animation signals
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
    )

    click = periodic_table(
        tiles=tiles,
        legend={
            "none": "#E5E7EB",
            "bad": "#EF4444",
            "close": "#F59E0B",
            "correct": "#16A34A",
            "hint": "#3B82F6",
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

    # Click handling
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

            # Normal + Hard: invalid guess doesn't consume a guess; shake + message; keep clues visible
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

    # Revealed clues — numbered, chronological order
    if st.session_state.revealed or special_hi_lo:
        st.subheader("Revealed clues")
        i = 1
        for k in st.session_state.revealed_order:
            if k in st.session_state.revealed:
                st.write(f"{i}. **{CLUES[k]}:** {st.session_state.revealed[k]}")
                i += 1
        if special_hi_lo:
            st.write(f"{i}. **Special clue:** {special_hi_lo}")

    if st.session_state.guesses:
        st.subheader("Guesses")
        labels = [f"{by_atomic[a].name} ({by_atomic[a].symbol})" for a in st.session_state.guesses]
        st.write(" • ".join(labels))

    # Share link button (small upgrade)
    st.subheader("🔗 Share this game")
    copy_link_button()
    st.caption("Tip: You can also copy the URL from your browser’s address bar.")

    # End states + Fun fact + share results
    if st.session_state.status in ("won", "lost"):
        if st.session_state.status == "won":
            st.success(f"🎉 Correct! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")
        else:
            st.error(f"😅 Out of guesses! It was **{secret.name} ({secret.symbol})**, atomic #{secret.atomic_number}.")

        st.subheader("Did you know?")
        if secret.discovered_by or secret.named_by:
            bits = []
            if secret.discovered_by:
                bits.append(f"**Discovered by:** {secret.discovered_by}")
            if secret.named_by:
                bits.append(f"**Named by:** {secret.named_by}")
            st.markdown("  \n".join(bits))

        if secret.summary:
            s = secret.summary.strip()
            if len(s) > 520:
                s = s[:520].rsplit(" ", 1)[0] + "…"
            st.write(s)

        if secret.source:
            st.caption(f"Source: {secret.source}")

        st.subheader("📋 Share results")
        share_text = build_share_text()
        copy_to_clipboard_button(share_text, label="📋 Copy results")
        with st.expander("Preview share text", expanded=False):
            st.code(share_text)
    else:
        st.caption("Finish the game to unlock Share Results + Fun Fact.")

    # Footer (small upgrade)
    st.markdown("---")
    st.caption(f"🧪 Element Guess • {APP_VERSION} • GitHub: {GITHUB_URL}")


if __name__ == "__main__":
    main()

