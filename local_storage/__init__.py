import os
import streamlit.components.v1 as components

# Optional dev server support if you ever want it
_DEV_URL = os.environ.get("LOCAL_STORAGE_DEV_URL", "")

if _DEV_URL:
    _component = components.declare_component("local_storage", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("local_storage", path=build_dir)


def local_storage(op: str, key: str, value=None, default=None):
    """
    op: "get" | "set"
    key: localStorage key
    value: JSON-serializable (for set)
    default: returned when missing/parse fails
    """
    return _component(op=op, key=key, value=value, default=default, default_value=default)
