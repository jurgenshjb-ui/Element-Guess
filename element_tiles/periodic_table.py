import os
import streamlit.components.v1 as components

# Optional dev server support (Vite dev)
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
    key="element_tiles",
):
    """
    tiles: list[dict] describing each tile
    legend: dict of status->color
    disabled: bool (locks clicks)
    invalidAtomic: int | None (optional, triggers shake styling in frontend)
    key: Streamlit widget key
    """
    return _component(
        tiles=tiles,
        legend=legend,
        disabled=disabled,
        invalidAtomic=invalidAtomic,
        key=key,
        default=None,
    )
