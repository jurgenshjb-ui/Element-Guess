import os
import streamlit.components.v1 as components

_DEV_URL = os.environ.get("ELEMENT_TILES_DEV_URL", "")

if _DEV_URL:
    _component = components.declare_component("element_tiles", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("element_tiles", path=build_dir)

def periodic_table(tiles, legend, disabled, key="element_tiles"):
    return _component(tiles=tiles, legend=legend, disabled=disabled, key=key, default=None)
