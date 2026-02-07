import os
import streamlit.components.v1 as components

# Optional dev server support if you ever want it
_DEV_URL = os.environ.get("LOCAL_STORAGE_DEV_URL", "").strip()

if _DEV_URL:
    _component = components.declare_component("local_storage", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("local_storage", path=build_dir)


def local_storage(op: str, storage_key: str, value=None, default=None, *, widget_key: str | None = None):
    """
    Minimal localStorage bridge.

    Args:
      op: "get" | "set"
      storage_key: localStorage key string (sent to the frontend)
      value: JSON-serializable value (for "set")
      default: returned when missing/parse fails
      widget_key: Streamlit widget key to keep calls distinct across reruns

    Returns:
      For "get": parsed JSON value (typically dict) or default
      For "set": {"ok": true} or {"ok": false, "error": "..."}
    """
    # IMPORTANT:
    # Streamlit reserves the kwarg name `key` for the widget identity.
    # So we send the actual localStorage key under `storage_key`.
    kwargs = dict(op=op, storage_key=storage_key, value=value, default=default)
    if widget_key is not None:
        kwargs["key"] = widget_key  # Streamlit widget key (NOT forwarded to frontend)
    return _component(**kwargs)
