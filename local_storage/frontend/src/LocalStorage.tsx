import React, { useEffect } from "react";
import { Streamlit, ComponentProps } from "streamlit-component-lib";

function safeParse(raw: string | null): any {
  if (raw == null) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export default function LocalStorage(props: ComponentProps) {
  const op: string = props.args["op"] ?? "get";

  // IMPORTANT: this must match app.py (storage_key, not key)
  const storageKey: string = props.args["storage_key"] ?? "";

  const value: any = props.args["value"] ?? null;
  const fallback: any = props.args["default"] ?? null;

  // Minimal height — we don't render UI
  useEffect(() => {
    Streamlit.setFrameHeight(0);
  }, []);

  useEffect(() => {
    if (!storageKey) {
      Streamlit.setComponentValue(fallback);
      return;
    }

    if (op === "get") {
      const raw = window.localStorage.getItem(storageKey);
      const parsed = safeParse(raw);
      Streamlit.setComponentValue(parsed ?? fallback);
      return;
    }

    if (op === "set") {
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(value));
        Streamlit.setComponentValue({ ok: true });
      } catch (e: any) {
        Streamlit.setComponentValue({ ok: false, error: String(e) });
      }
      return;
    }

    // Unknown op
    Streamlit.setComponentValue(fallback);
  }, [op, storageKey, value, fallback]);

  return null;
}
