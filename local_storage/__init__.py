import os
import streamlit.components.v1 as components

_DEV_URL = os.environ.get("LOCAL_STORAGE_DEV_URL", "").strip()

if _DEV_URL:
    _component = components.declare_component("local_storage", url=_DEV_URL)
else:
    build_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    _component = components.declare_component("local_storage", path=build_dir)


def local_storage(op: str, storage_key: str, value=None, default=None, *, widget_key: str | None = None):
    """
    op: "get" | "set"
    storage_key: browser localStorage key (sent to JS)
    widget_key: Streamlit widget key (NOT sent to JS)
    """
    kwargs = dict(op=op, storage_key=storage_key, value=value, default=default)
    if widget_key is not None:
        kwargs["key"] = widget_key
    return _component(**kwargs)
