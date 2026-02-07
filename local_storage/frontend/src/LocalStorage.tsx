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
  const args = props.args ?? {};

  const op: string = (args["op"] ?? "get") as string;
  const storageKey: string = (args["key"] ?? "") as string;

  const value: any = args["value"] ?? null;

  // Support both names because your Python calls have varied:
  const fallback: any =
    args["default_value"] ?? args["default"] ?? props.defaultValue ?? null;

  // Tell Streamlit the component is ready + keep it zero-height
  useEffect(() => {
    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(0);
  }, []);

  useEffect(() => {
    Streamlit.setFrameHeight(0);

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
  }, [op, storageKey, value, JSON.stringify(fallback)]);

  return null;
}
