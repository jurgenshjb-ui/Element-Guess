import React from "react";
import ReactDOM from "react-dom/client";
import LocalStorage from "./LocalStorage";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <LocalStorage />
  </React.StrictMode>
);
