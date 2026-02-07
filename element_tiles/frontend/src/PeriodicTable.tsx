import React, { useEffect, useMemo } from "react";
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
  s: string;
  p: string;
  d: string;
  f: string;
};

function fgColor(status: Tile["status"]) {
  if (status === "lost") return "#FDE68A";
  return status === "bad" || status === "correct" ? "white" : "#111827";
}

function blockFor(t: Tile): "s" | "p" | "d" | "f" | "u" {
  if (t.row === "lanth" || t.row === "actin") return "f";
  const g = t.group ?? null;
  if (g === null) return "u";
  if (g === 1 || g === 2 || t.symbol === "H" || t.symbol === "He") return "s";
  if (g >= 3 && g <= 12) return "d";
  if (g >= 13 && g <= 18) return "p";
  return "u";
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
    s: "#60A5FA",
    p: "#FBBF24",
    d: "#34D399",
    f: "#A78BFA",
  };
  const disabled: boolean = !!props.args["disabled"];

  // Build main grid lookup for block-edge calculation
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

  const edgeMap = useMemo(() => {
    const edges = new Map<number, {top:boolean;right:boolean;bottom:boolean;left:boolean}>();
    const get = (p:number,g:number) => main.get(`${p}-${g}`);
    for (let p=1; p<=7; p++){
      for (let g=1; g<=18; g++){
        const t = get(p,g);
        if (!t) continue;
        const b = blockFor(t);
        if (b==="u") continue;
        const nTop = get(p-1,g);
        const nBot = get(p+1,g);
        const nLeft = get(p,g-1);
        const nRight = get(p,g+1);
        edges.set(t.atomic,{
          top: !(nTop && blockFor(nTop)===b),
          bottom: !(nBot && blockFor(nBot)===b),
          left: !(nLeft && blockFor(nLeft)===b),
          right: !(nRight && blockFor(nRight)===b),
        });
      }
    }
    // strips: treat contiguous within strip
    const stripEdges = (strip:Tile[])=>{
      for (let i=0;i<strip.length;i++){
        const t=strip[i];
        const b=blockFor(t);
        if (b!=="f") continue;
        const l = strip[i-1];
        const r = strip[i+1];
        edges.set(t.atomic,{
          top:true,
          bottom:true,
          left: !(l && blockFor(l)===b),
          right: !(r && blockFor(r)===b),
        });
      }
    };
    stripEdges(lanth);
    stripEdges(actin);
    return edges;
  }, [tiles]);

  const sendClick = (t: Tile) => {
    if (disabled) return;
    if (t.locked) return;
    Streamlit.setComponentValue({ atomic: t.atomic, nonce: Math.floor(Math.random() * 1e9) });
  };

  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  const cellW = 38;
  const cellH = 32;

  const renderCell = (t?: Tile) => {
    if (!t) return <td style={{ width: cellW, height: cellH }} />;

    const b = blockFor(t);
    const e = edgeMap.get(t.atomic);
    const cls =
      "tile" +
      (t.status ? ` ${t.status}` : "") +
      ((disabled || t.locked) ? " disabled" : "") +
      (t.isNewHint ? " pulse" : "") +
      (t.isLastGuess ? " lastGuess" : "") +
      (t.isWin ? " win" : "") +
      (b !== "u" ? ` block-${b}` : "") +
      (e?.top ? " edge-top" : "") +
      (e?.right ? " edge-right" : "") +
      (e?.bottom ? " edge-bottom" : "") +
      (e?.left ? " edge-left" : "");

    return (
      <td style={{ width: cellW, height: cellH, textAlign: "center" }}>
        <div
          className={cls}
          style={{ background: legend[t.status], color: fgColor(t.status) }}
          title={t.tooltip}
          onClick={() => sendClick(t)}
        >
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
          <td style={{ width: cellW, height: cellH }} />
          <td style={{ width: cellW, height: cellH }} />
          {strip.slice(0, 15).map((t) => renderCell(t))}
        </tr>
      </tbody>
    </table>
  );

  return (
    <div className="wrap">
      <div className="legend">
        <span className="chip"><span className="box" style={{ background: legend.none }} /> untried</span>
        <span className="chip"><span className="box" style={{ background: legend.bad }} /> fails clues</span>
        <span className="chip"><span className="box" style={{ background: legend.close }} /> matches clues</span>
        <span className="chip"><span className="box" style={{ background: legend.correct }} /> correct</span>
        <span className="chip"><span className="box" style={{ background: legend.hint }} /> easy candidates</span>
        <span className="chip"><span className="box" style={{ background: legend.lost }} /> revealed answer</span>
      </div>

      <div className="legend blockLegend">
        <span className="chip"><span className="box outline" style={{ borderColor: legend.s }} /> s‑block</span>
        <span className="chip"><span className="box outline" style={{ borderColor: legend.p }} /> p‑block</span>
        <span className="chip"><span className="box outline" style={{ borderColor: legend.d }} /> d‑block</span>
        <span className="chip"><span className="box outline" style={{ borderColor: legend.f }} /> f‑block</span>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th className="th-left"></th>
            {Array.from({ length: 18 }, (_, i) => (
              <th key={i} className="th">{i + 1}</th>
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
