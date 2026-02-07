import os
import streamlit.components.v1 as components

_DEV_URL = os.environ.get("ELEMENT_TILES_DEV_URL", "")

if _DEV_URL:
    _component = components.declare_component("element_tiles", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("element_tiles", path=build_dir)

def periodic_table(
    tiles,
    legend,
    disabled=False,
    invalidAtomic=None,
    blockLegend=None,
    key="element_tiles",
):
    """Render the periodic table Streamlit component.

    Parameters
    ----------
    tiles : list[dict]
        Tile payloads.
    legend : dict
        Status colours.
    disabled : bool
        Disable clicking.
    invalidAtomic : int | None
        Atomic number to shake (invalid guess feedback).
    blockLegend : dict | None
        Colours for s/p/d/f block outlines and legend.
        Example: {"s": "#...", "p": "#...", "d": "#...", "f": "#..."}
    key : str
        Streamlit component key.
    """
    return _component(
        tiles=tiles,
        legend=legend,
        disabled=disabled,
        invalidAtomic=invalidAtomic,
        blockLegend=blockLegend,
        key=key,
        default=None,
    )
