import os
import streamlit.components.v1 as components

# When developing, you can point to a dev server
_DEV_URL = os.environ.get("ELEMENT_TILES_DEV_URL", "")

if _DEV_URL:
    _component = components.declare_component("element_tiles", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("element_tiles", path=build_dir)


def periodic_table(
    tiles: list[dict],
    legend: dict,
    disabled: bool,
    invalidAtomic: int | None = None,   #  NEW
    key: str = "element_tiles",
):
    """
    tiles: list of dicts describing each tile:
      {
        "symbol": "Br",
        "name": "Bromine",
        "atomic": 35,
        "group": 17,
        "period": 4,
        "row": "main" | "lanth" | "actin",
        "pos": int,             # position in lanth/actin row (0..14) if row != main
        "status": "none" | "bad" | "close" | "correct" | "hint",
        "tooltip": "...",

        # optional animation flags (if you send them)
        "isLastGuess": bool,
        "isNewHint": bool,
        "isWin": bool
      }

    Returns a dict like {"atomic": 35, "nonce": 12345} or None
    """
    return _component(
        tiles=tiles,
        legend=legend,
        disabled=disabled,
        invalidAtomic=invalidAtomic,     #  NEW (forward to frontend)
        key=key,
        default=None,
    )
