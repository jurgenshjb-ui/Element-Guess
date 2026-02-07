import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";

import PeriodicTable from "./PeriodicTable";
import "./styles.css";

// Wrap with Streamlit connection so props (args) arrive and the component
// signals readiness to Streamlit.
const ConnectedPeriodicTable = withStreamlitConnection(PeriodicTable);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ConnectedPeriodicTable />
  </React.StrictMode>
);
