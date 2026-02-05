import React, { useEffect } from "react";
import { Streamlit, ComponentProps } from "streamlit-component-lib";

type Tile = {
  symbol: string;
  name: string;
  atomic: number;
  group?: number | null;
  period?: number | null;
  row: "main" | "lanth" | "actin";
  pos?: number;
  status: "none" | "bad" | "close" | "correct" | "hint" | "lost";
  tooltip: string;
  locked?: boolean;

  // Animation flags (from app.py)
  isLastGuess?: boolean;
  isNewHint?: boolean;
  isWin?: boolean;
};

type Legend = {
  none: string;
  bad: string;
  close: string;
  correct: string;
  hint: string;
  lost: string;
};

function fgColor(status: Tile["status"]) {
  return status === "bad" || status === "correct" || status === "lost"
  ? "white"
  : "#111827";
}

const PeriodicTable = (props: ComponentProps) => {
  const tiles: Tile[] = props.args["tiles"] ?? [];
  const legend: Legend = props.args["legend"] ?? {
    none: "#E5E7EB",
    bad: "#EF4444",
    close: "#F59E0B",
    correct: "#16A34A",
    hint: "#3B82F6",
    lost: "#111827",
  };
  const disabled: boolean = !!props.args["disabled"];
  const invalidAtomic: number | null = props.args["invalidAtomic"] ?? null;

  // --- Build lookup tables
  const main = new Map<string, Tile>();
  const lanth: Tile[] = [];
  const actin: Tile[] = [];

  for (const t of tiles) {
    if (t.row === "main" && t.period && t.group) {
      main.set(`${t.period}-${t.group}`, t);
    } else if (t.row === "lanth") {
      lanth.push(t);
    } else if (t.row === "actin") {
      actin.push(t);
    }
  }

  lanth.sort((a, b) => (a.pos ?? 0) - (b.pos ?? 0));
  actin.sort((a, b) => (a.pos ?? 0) - (b.pos ?? 0));

  // --- Click handler
  const sendClick = (t: Tile) => {
    if (disabled) return;
    if (t.locked) return;

    Streamlit.setComponentValue({
      atomic: t.atomic,
      nonce: Math.floor(Math.random() * 1e9),
    });
  };

  // Let Streamlit resize iframe if needed
  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  const cellW = 38;
  const cellH = 32;

  const renderCell = (t?: Tile) => {
    if (!t) return <td style={{ width: cellW, height: cellH }} />;

    const shake = invalidAtomic != null && t.atomic === invalidAtomic;

    const cls =
      "tile" +
      (t.status ? ` ${t.status}` : "") +
      ((disabled || t.locked) ? " disabled" : "") +
      (t.isNewHint ? " pulse" : "") +
      (t.isLastGuess ? " lastGuess" : "") +
      (t.isWin ? " win" : "") +
      (shake ? " shake" : "");

    return (
      <td style={{ width: cellW, height: cellH, textAlign: "center" }}>
        <div
          className={cls}
          style={{
            background: legend[t.status],
            color: fgColor(t.status),
          }}
          title={t.tooltip}
          onClick={() => sendClick(t)}
        >
          {/* Atomic number + symbol (stacked) */}
          <div className="anum">{t.atomic}</div>
          <div className="sym">{t.symbol}</div>
        </div>
      </td>
    );
  };

  const stripRow = (label: string, strip: Tile[]) => (
    <table className="table" style={{ marginTop: 6 }}>
      <tbody>
        <tr>
          <th className="stripLabel">{label}</th>
          {/* spacer for group 1 & 2 alignment */}
          <td style={{ width: cellW, height: cellH }} />
          <td style={{ width: cellW, height: cellH }} />
          {strip.slice(0, 15).map((t) => renderCell(t))}
        </tr>
      </tbody>
    </table>
  );

  return (
    <div className="wrap">
      {/* Legend */}
      <div className="legend">
        <span className="chip">
          <span className="box" style={{ background: legend.none }} /> untried
        </span>
        <span className="chip">
          <span className="box" style={{ background: legend.bad }} /> fails clues
        </span>
        <span className="chip">
          <span className="box" style={{ background: legend.close }} /> matches clues
        </span>
        <span className="chip">
          <span className="box" style={{ background: legend.correct }} /> correct
        </span>
        <span className="chip">
          <span className="box" style={{ background: legend.hint }} /> easy candidates
        </span>
        <span className="chip">
          <span className="box" style={{ background: legend.lost }} /> revealed answer
        </span>
      </div>

      {/* Main periodic table */}
      <table className="table">
        <thead>
          <tr>
            <th className="th-left"></th>
            {Array.from({ length: 18 }, (_, i) => (
              <th key={i} className="th">
                {i + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: 7 }, (_, pi) => {
            const period = pi + 1;
            return (
              <tr key={period}>
                <th className="th-left">{period}</th>
                {Array.from({ length: 18 }, (_, gi) => {
                  const group = gi + 1;
                  const t = main.get(`${period}-${group}`);
                  return renderCell(t);
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="hr"></div>

      {stripRow("Lanthanoids (57–71)", lanth)}
      {stripRow("Actinoids (89–103)", actin)}
    </div>
  );
};

export default PeriodicTable;
