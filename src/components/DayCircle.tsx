import { useRef } from "react";

interface Props {
  days: string[];
  /** measured anchor angle (radians) per day. */
  angles: number[];
  /** current dial angle (radians). */
  theta: number;
  /** the residual's natural angle for this prompt (faint marker). */
  baselineAngle: number;
  /** predicted next-day index → highlighted. */
  topDay: number;
  /** softmax over days, used to scale anchor glow. */
  dayProbs: number[];
  size?: number;
  onTheta: (theta: number) => void;
}

// math-convention angle → screen point (y flipped)
const pt = (cx: number, cy: number, r: number, a: number): [number, number] => [
  cx + r * Math.cos(a),
  cy - r * Math.sin(a),
];

export default function DayCircle({
  days, angles, theta, baselineAngle, topDay, dayProbs, size = 460, onTheta,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const S = size, cx = S / 2, cy = S / 2;
  const R = S * 0.34;        // ring radius
  const Rlabel = S * 0.43;   // label radius

  const angleFromEvent = (e: { clientX: number; clientY: number }) => {
    const el = svgRef.current;
    if (!el) return theta;
    const rect = el.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * S - cx;
    const y = ((e.clientY - rect.top) / rect.height) * S - cy;
    return Math.atan2(-y, x);
  };

  const start = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    onTheta(angleFromEvent(e));
  };
  const move = (e: React.PointerEvent) => { if (dragging.current) onTheta(angleFromEvent(e)); };
  const end = () => { dragging.current = false; };

  const [hx, hy] = pt(cx, cy, R, theta);
  const [bx, by] = pt(cx, cy, R, baselineAngle);

  // succession arc from dial angle → predicted-day anchor
  const tgt = angles[topDay] ?? theta;
  let delta = tgt - theta;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  const arcR = R * 0.7;
  const [ax0, ay0] = pt(cx, cy, arcR, theta);
  const [ax1, ay1] = pt(cx, cy, arcR, tgt);
  const largeArc = Math.abs(delta) > Math.PI ? 1 : 0;
  const sweep = delta > 0 ? 0 : 1; // screen y flipped → invert sweep
  const arcPath = `M ${ax0} ${ay0} A ${arcR} ${arcR} 0 ${largeArc} ${sweep} ${ax1} ${ay1}`;
  const [headx, heady] = [ax1, ay1];

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${S} ${S}`}
      width="100%"
      style={{ touchAction: "none", cursor: dragging.current ? "grabbing" : "grab", maxWidth: S }}
      onPointerDown={start}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
    >
      <defs>
        <radialGradient id="dc-core" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(211,90,75,0.16)" />
          <stop offset="70%" stopColor="rgba(211,90,75,0.03)" />
          <stop offset="100%" stopColor="rgba(211,90,75,0)" />
        </radialGradient>
      </defs>

      <circle cx={cx} cy={cy} r={R * 1.32} fill="url(#dc-core)" />
      {/* the ring */}
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--canvas-rule-2)" strokeWidth={1} />

      {/* succession arc: "you are here → it predicts one hop around" */}
      <path d={arcPath} fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeLinecap="round" opacity={0.85} />
      <circle cx={headx} cy={heady} r={4} fill="var(--accent)" />

      {/* spoke to the dial handle */}
      <line x1={cx} y1={cy} x2={hx} y2={hy} stroke="var(--accent)" strokeWidth={1} opacity={0.45} />

      {/* day anchors + labels */}
      {days.map((d, i) => {
        const [px, py] = pt(cx, cy, R, angles[i]);
        const [lx, ly] = pt(cx, cy, Rlabel, angles[i]);
        const isTop = i === topDay;
        const glow = 3 + (dayProbs[i] ?? 0) * 9;
        return (
          <g key={d}>
            {isTop && <circle cx={px} cy={py} r={glow} fill="var(--accent)" opacity={0.22} />}
            <circle
              cx={px} cy={py} r={isTop ? 6 : 3.5}
              fill={isTop ? "var(--accent)" : "var(--canvas-ink-faint)"}
            />
            <text
              x={lx} y={ly}
              textAnchor="middle" dominantBaseline="middle"
              className="mono"
              style={{
                fontSize: 12,
                fill: isTop ? "var(--accent)" : "var(--canvas-ink-dim)",
                fontWeight: isTop ? 600 : 400,
                letterSpacing: "0.02em",
              }}
            >
              {d.slice(0, 3)}
            </text>
          </g>
        );
      })}

      {/* baseline (natural) marker */}
      <circle cx={bx} cy={by} r={5} fill="none" stroke="var(--canvas-ink-faint)" strokeWidth={1.5} strokeDasharray="2 2" />

      {/* the draggable dial handle */}
      <circle cx={hx} cy={hy} r={9} fill="var(--accent)" stroke="var(--canvas)" strokeWidth={2} />
      <circle cx={hx} cy={hy} r={15} fill="none" stroke="var(--accent)" strokeWidth={1} opacity={0.3} />
    </svg>
  );
}
