import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import PeriodicTable from "./PeriodicTable";
import "./styles.css";

const Connected = withStreamlitConnection(PeriodicTable);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Connected />
  </React.StrictMode>
);
