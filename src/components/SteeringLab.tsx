import { useCallback, useEffect, useRef, useState } from "react";
import { encode } from "gpt-tokenizer/esm/encoding/r50k_base";
import type { SteerConstants, PromptDef, Prediction } from "../lib/steering/types";
import { initSteering, runPre, runPost, type LoadProgress } from "../lib/steering/onnx";
import { rotateResidual, ablateResidual, projectAngle, readPrediction } from "../lib/steering/intervention";
import DayCircle from "./DayCircle";

type Active = { text: string; input_ids: number[] };
const GPT2_BOS = 50256;
const MAX_STEER = 6;

const PAPER_URL = "https://arxiv.org/abs/2405.14860";
const CODE_URL = "https://github.com/APatelUIUC/rotating-week";
const AUTHOR_URL = "https://www.akashpa.tel";

const mod7 = (x: number) => ((x % 7) + 7) % 7;
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const nearestDay = (theta: number, angles: number[]) => {
  let best = 0, bd = Infinity;
  for (let i = 0; i < angles.length; i++) {
    let d = Math.abs(theta - angles[i]);
    d = Math.min(d, 2 * Math.PI - d);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
};

export default function SteeringLab() {
  const [K, setK] = useState<SteerConstants | null>(null);
  const [prompts, setPrompts] = useState<PromptDef[]>([]);
  const [load, setLoad] = useState<LoadProgress | null>(null);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [active, setActive] = useState<Active | null>(null);
  const [customText, setCustomText] = useState("");
  const [steer, setSteer] = useState(0);          // accumulated steering, in days
  const [offPlane, setOffPlane] = useState(0);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [baselinePred, setBaselinePred] = useState<Prediction | null>(null);
  const [baselineAngle, setBaselineAngle] = useState(0);
  const [ablation, setAblation] = useState<{ kind: "circle" | "random"; pred: Prediction } | null>(null);

  const residRef = useRef<{ resid: Float32Array; seq: number; dModel: number } | null>(null);
  const job = useRef<{ running: boolean; pending: { angle: number | null; off: number } | null }>({ running: false, pending: null });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [kc, pj] = await Promise.all([
          fetch("/steering/constants.json").then((r) => r.json()),
          fetch("/steering/prompts.json").then((r) => r.json()),
        ]);
        if (!alive) return;
        setK(kc); setPrompts(pj.prompts);
        await initSteering((p) => alive && setLoad(p));
        if (!alive) return;
        setReady(true);
      } catch (e) { if (alive) setErr(String(e)); }
    })();
    return () => { alive = false; };
  }, []);

  const K_ = K;

  // Run the decoder. angle === null → pristine residual (the natural state, no
  // intervention); otherwise rotate the in-plane component to `angle`.
  const decodeAt = useCallback(async (angle: number | null, off: number) => {
    if (!K_ || !residRef.current) return;
    if (job.current.running) { job.current.pending = { angle, off }; return; }
    job.current.running = true;
    try {
      const { resid, seq, dModel } = residRef.current;
      const r = angle === null ? resid : rotateResidual(resid, seq, dModel, K_, angle, off);
      setPred(readPrediction(await runPost(r, seq, dModel), K_));
    } finally {
      job.current.running = false;
      const p = job.current.pending;
      if (p) { job.current.pending = null; decodeAt(p.angle, p.off); }
    }
  }, [K_]);

  useEffect(() => { if (ready && prompts.length && !active) setActive(prompts[0]); }, [ready, prompts, active]);

  // (re)encode + reset to the natural state whenever the prompt changes
  useEffect(() => {
    if (!K_ || !ready || !active) return;
    let alive = true;
    (async () => {
      const enc = await runPre(active.input_ids);
      if (!alive) return;
      residRef.current = enc;
      const { angle } = projectAngle(enc.resid, enc.seq, enc.dModel, K_);
      setBaselineAngle(angle);
      setSteer(0);
      setAblation(null);
      const natural = readPrediction(await runPost(enc.resid, enc.seq, enc.dModel), K_);
      if (!alive) return;
      setBaselinePred(natural);
      setPred(natural);
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ready]);

  if (err) return <Shell><div className="mono text-[var(--accent)] p-6">{err}</div></Shell>;
  if (!K_ || !ready || !active) return <Shell><Loader load={load} /></Shell>;

  const angles = K_.day_angles_rad;
  const baselineDay = nearestDay(baselineAngle, angles);
  const internalDay = mod7(baselineDay + steer);     // day the model is rotated to read
  const theta = angles[internalDay];                  // dial position (derived from steer)
  const refDay = baselinePred?.topDay ?? 0;           // the natural next-word
  const predDay = pred?.topDay ?? refDay;             // the current next-word
  const selectedCurated = prompts.findIndex((p) => p.text === active.text);
  const overallIsDay = pred ? K_.days.some((d) => pred.overallTop.trim() === d) : true;

  const sgn7 = (x: number) => { let d = mod7(x); if (d > 3) d -= 7; return d; };
  const predShift = sgn7(predDay - refDay);
  const fmt = (n: number) => (n > 0 ? `+${n}` : `${n}`);
  const steerLabel = (n: number) => `${n > 0 ? "+" : "−"}${Math.abs(n)} day${Math.abs(n) === 1 ? "" : "s"}`;
  const tracking = predShift === steer;

  // apply a new steer value: angle is null at 0 (pristine), else the target anchor
  const applySteer = (s: number) => {
    const next = clamp(s, -MAX_STEER, MAX_STEER);
    setSteer(next);
    setAblation(null);
    decodeAt(next === 0 ? null : angles[mod7(baselineDay + next)], offPlane);
  };
  // drag → snap to the nearest day, picking the lap that's closest to the current steer
  const onDragAngle = (a: number) => {
    const dd = nearestDay(a, angles);
    let best = steer, bd = Infinity;
    for (let k = -MAX_STEER; k <= MAX_STEER; k++) {
      if (mod7(baselineDay + k) === dd) { const d = Math.abs(k - steer); if (d < bd) { bd = d; best = k; } }
    }
    applySteer(best);
  };
  const onOff = (off: number) => { setOffPlane(off); decodeAt(steer === 0 ? null : angles[internalDay], off); };
  const runAblation = async (kind: "circle" | "random") => {
    if (!residRef.current) return;
    const { resid, seq, dModel } = residRef.current;
    const r = ablateResidual(resid, seq, dModel, K_, kind);
    setAblation({ kind, pred: readPrediction(await runPost(r, seq, dModel), K_) });
  };
  const submitCustom = () => {
    const t = customText.trim();
    if (!t) return;
    setActive({ text: t, input_ids: [GPT2_BOS, ...encode(t)].slice(0, 64) });
  };

  return (
    <Shell>
      <div className="mx-auto max-w-[1180px] px-6 pb-24">
        <header className="pt-8 pb-8 max-w-[760px]">
          <div className="mono text-[12px] tracking-[0.18em] uppercase text-[var(--ink-faint)] mb-4">
            Live circuit · GPT-2 small
          </div>
          <h1 className="h1-display mb-4" style={{ fontSize: "clamp(40px,5vw,68px)" }}>
            Rotating the&nbsp;week
          </h1>
          <p className="serif text-[19px] leading-[1.5] text-[var(--ink-2)]">
            GPT-2 small stores the days of the week on a <em>circle</em> in its residual
            stream, and predicts the next day by <em>rotating around it</em>. Steer that circle
            by a few days and the model's next-word prediction moves by the same amount — a
            causal handle on how it computes succession.
          </p>
        </header>

        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_360px] items-start">
          <figure
            className="rounded-[14px] p-6 relative"
            style={{ background: "var(--canvas)", boxShadow: "0 1px 0 var(--paper-rule) inset, 0 18px 50px -28px rgba(26,22,18,0.5)" }}
          >
            <figcaption className="flex items-center justify-between gap-3 mb-2">
              <span className="mono text-[11px] tracking-[0.16em] uppercase text-[var(--canvas-ink-faint)] truncate min-w-0">
                blocks.{K_.layer}.hook_resid_pre
              </span>
              <span className="mono text-[11px] text-[var(--canvas-ink-faint)] whitespace-nowrap shrink-0">
                {offPlane === 0 ? "isolated circle" : offPlane === 1 ? "full residual" : `off-plane ×${offPlane.toFixed(2)}`}
              </span>
            </figcaption>

            <div className="rounded-[8px] px-3 py-2 mb-1" style={{ border: "1px solid var(--canvas-rule-2)" }}>
              <div className="mono text-[10px] tracking-[0.14em] uppercase text-[var(--canvas-ink-faint)] mb-1">
                prompt · the text never changes
              </div>
              <div className="mono text-[12.5px] text-[var(--canvas-ink-dim)] break-words">{active.text}</div>
            </div>

            <div className="flex justify-center py-1">
              <DayCircle
                days={K_.days}
                angles={angles}
                theta={theta}
                baselineAngle={baselineAngle}
                topDay={predDay}
                dayProbs={pred?.dayProbs ?? K_.days.map(() => 0)}
                onTheta={onDragAngle}
              />
            </div>

            <p className="text-center mono text-[10px] tracking-[0.12em] uppercase text-[var(--canvas-ink-faint)] -mt-1 mb-3">
              drag the handle — or use −/+ — to rotate GPT-2's internal day
            </p>

            {/* the steering counter — the intervention magnitude */}
            <div className="flex items-center justify-center gap-5 mb-4">
              <RoundBtn onClick={() => applySteer(steer - 1)} disabled={steer <= -MAX_STEER}>−</RoundBtn>
              <div className="text-center min-w-[140px]">
                <div className="mono text-[10px] tracking-[0.16em] uppercase text-[var(--canvas-ink-faint)]">steering</div>
                <div className="serif text-[26px] leading-tight" style={{ color: steer === 0 ? "var(--canvas-ink-dim)" : "var(--accent-soft)" }}>
                  {steer === 0 ? "natural" : `${steerLabel(steer)}`}
                </div>
              </div>
              <RoundBtn onClick={() => applySteer(steer + 1)} disabled={steer >= MAX_STEER}>+</RoundBtn>
            </div>

            {/* natural → steered prediction */}
            <div className="text-center">
              {steer === 0 ? (
                <p className="mono text-[11px] text-[var(--canvas-ink-faint)] leading-relaxed">
                  GPT-2's natural next word is{" "}
                  <span className="serif text-[18px] text-[var(--canvas-ink)] align-middle">{K_.days[refDay]}</span>
                  {" "}· steer the circle to push it around the week
                </p>
              ) : (
                <>
                  <div className="mono text-[10px] tracking-[0.14em] uppercase text-[var(--canvas-ink-faint)]">
                    natural was {K_.days[refDay]} — now GPT-2 predicts
                  </div>
                  <div className="serif text-[30px] leading-tight" style={{ color: "var(--accent-soft)" }}>{K_.days[predDay]}</div>
                  <div className="mono text-[11px] mt-1" style={{ color: tracking ? "var(--accent-soft)" : "var(--canvas-ink-faint)" }}>
                    {tracking
                      ? `▸ steered ${steerLabel(steer)} · prediction moved ${fmt(predShift)} — in lockstep`
                      : `▸ steered ${steerLabel(steer)} · expected ${K_.days[mod7(refDay + steer)]}, got ${K_.days[predDay]} — the weekend seam`}
                  </div>
                  <p className="mt-3 serif text-[13px] italic text-[var(--canvas-ink-dim)] max-w-[430px] mx-auto leading-snug">
                    The words never changed — you rotated a circle inside layer {K_.layer}, and the
                    prediction followed.
                  </p>
                </>
              )}
            </div>
          </figure>

          <aside className="space-y-7">
            <Field label="Prompt">
              <div className="space-y-2">
                {prompts.map((p, i) => (
                  <button
                    key={p.text}
                    onClick={() => setActive(p)}
                    className="w-full text-left rounded-[8px] px-3 py-2 transition-colors mono text-[12.5px]"
                    style={{
                      background: i === selectedCurated ? "var(--accent-tint)" : "transparent",
                      border: `1px solid ${i === selectedCurated ? "var(--accent)" : "var(--paper-rule-2)"}`,
                      color: i === selectedCurated ? "var(--ink)" : "var(--ink-dim)",
                    }}
                  >
                    {p.text}
                  </button>
                ))}
                <form onSubmit={(e) => { e.preventDefault(); submitCustom(); }} className="flex gap-2 pt-1">
                  <input
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    placeholder="…or type your own"
                    className="flex-1 rounded-[8px] px-3 py-2 mono text-[12.5px] bg-transparent outline-none"
                    style={{ border: `1px solid ${selectedCurated < 0 ? "var(--accent)" : "var(--paper-rule-2)"}`, color: "var(--ink)" }}
                  />
                  <button type="submit" className="rounded-[8px] px-3 mono text-[12px]"
                          style={{ border: "1px solid var(--accent)", color: "var(--accent)" }}>run</button>
                </form>
                {selectedCurated < 0 && (
                  <p className="mono text-[10.5px] text-[var(--ink-faint)] leading-snug">
                    custom prompt — steering still acts on the final token, but the effect is
                    cleanest on day-sequence prompts.
                  </p>
                )}
              </div>
            </Field>

            <Field label="Predicted next word">
              <Bars days={K_.days} probs={pred?.dayProbs ?? K_.days.map(() => 0)} top={predDay} />
              {pred && !overallIsDay && (
                <p className="mono text-[11px] text-[var(--ink-faint)] mt-2 leading-snug">
                  top token overall: “{pred.overallTop}” ({(pred.overallProb * 100).toFixed(0)}%) — not a day
                </p>
              )}
            </Field>

            <Field label={
              <span className="flex items-center justify-between">
                <span>Off-plane signal</span>
                <span className="mono text-[10px] normal-case tracking-normal text-[var(--ink-faint)]">
                  {offPlane === 0 ? "isolate" : offPlane === 1 ? "preserve" : offPlane.toFixed(2)}
                </span>
              </span>
            }>
              <input type="range" min={0} max={1.5} step={0.05} value={offPlane}
                     onChange={(e) => onOff(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
              <p className="text-[12px] text-[var(--ink-dim)] leading-snug mt-2">
                Steering only ever touches the 2-D day-circle. This sets how much of the
                residual's <em>off-circle</em> content rides along: <strong>isolate&nbsp;(0)</strong> drops the
                competing signal for a crisp result; <strong>preserve&nbsp;(1)</strong> is the untouched
                residual. Push past 1 to amplify and watch it break.
              </p>
            </Field>

            <Field label="Is the circle necessary?">
              <p className="text-[12px] text-[var(--ink-dim)] leading-snug mb-3">
                Steering shows the circle can <em>move</em> the answer (sufficiency). This removes
                a 2-D subspace and re-runs the model: kill the day-circle and the prediction
                should break — kill a random plane of the same size and it shouldn't.
              </p>
              <div className="flex gap-2 mb-3">
                <Btn onClick={() => runAblation("circle")} subtle={ablation?.kind !== "circle"}>remove day-circle</Btn>
                <Btn onClick={() => runAblation("random")} subtle={ablation?.kind !== "random"}>remove random plane</Btn>
              </div>
              {ablation && baselinePred && (
                <div className="rounded-[8px] p-3" style={{ background: "var(--paper-2)", border: "1px solid var(--paper-rule-2)" }}>
                  <div className="mono text-[11px] text-[var(--ink-dim)] mb-1">
                    natural answer: <span style={{ color: "var(--ink)" }}>{K_.days[refDay]}</span>{" "}
                    ({((baselinePred.dayProbs[refDay] ?? 0) * 100).toFixed(0)}%)
                  </div>
                  <div className="mono text-[12px]" style={{ color: ablation.kind === "circle" ? "var(--accent)" : "var(--ink)" }}>
                    {ablation.kind === "circle" ? "day-circle removed" : "random plane removed"} → now{" "}
                    <span style={{ fontWeight: 600 }}>{K_.days[ablation.pred.topDay]}</span>{" "}
                    ({((ablation.pred.dayProbs[refDay] ?? 0) * 100).toFixed(0)}% on {K_.days[refDay].slice(0, 3)})
                  </div>
                  <p className="text-[11px] text-[var(--ink-dim)] leading-snug mt-2">
                    {ablation.kind === "circle"
                      ? "the circle is carrying the succession signal — remove it and the model loses the answer."
                      : "removing an unrelated plane of the same size barely moves it — the effect is specific to the circle."}
                  </p>
                </div>
              )}
            </Field>
          </aside>
        </div>

        <MethodsPanel K={K_} />
      </div>
    </Shell>
  );
}

/* ---------- presentational ---------- */

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full" style={{ background: "var(--paper)" }}>
      <nav className="mx-auto max-w-[1180px] px-6 pt-6 flex items-center justify-between">
        <span className="mono text-[12px] tracking-[0.14em] uppercase text-[var(--ink-2)]">
          <span style={{ color: "var(--accent)" }}>●</span> Rotating the Week
        </span>
        <div className="flex items-center gap-4 mono text-[11.5px] text-[var(--ink-dim)]">
          <a href={PAPER_URL} target="_blank" rel="noreferrer" className="hover:text-[var(--accent)] transition-colors">paper ↗</a>
          <a href={CODE_URL} target="_blank" rel="noreferrer" className="hover:text-[var(--accent)] transition-colors">code ↗</a>
          <a href={AUTHOR_URL} target="_blank" rel="noreferrer" className="hover:text-[var(--accent)] transition-colors">Akash&nbsp;Patel ↗</a>
        </div>
      </nav>
      {children}
    </div>
  );
}

function Loader({ load }: { load: LoadProgress | null }) {
  const pct = Math.round((load?.frac ?? 0) * 100);
  return (
    <div className="mx-auto max-w-[560px] px-6 py-32 text-center">
      <h1 className="serif text-[28px] text-[var(--ink)] mb-2">Loading GPT-2 into your browser</h1>
      <p className="text-[14px] text-[var(--ink-dim)] mb-8">
        {load?.stage ?? "Fetching constants"} · {load?.receivedMB ?? 0}/{load?.totalMB ?? 169} MB
      </p>
      <div className="h-[3px] rounded-full overflow-hidden" style={{ background: "var(--paper-rule-2)" }}>
        <div className="h-full transition-[width] duration-200" style={{ width: `${pct}%`, background: "var(--accent)" }} />
      </div>
      <p className="mono text-[11px] text-[var(--ink-faint)] mt-3">{pct}% · one-time download, then cached by your browser</p>
    </div>
  );
}

function Field({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <div className="mono text-[11px] tracking-[0.14em] uppercase text-[var(--ink-faint)] mb-3">{label}</div>
      {children}
    </div>
  );
}

function Btn({ children, onClick, subtle }: { children: React.ReactNode; onClick: () => void; subtle?: boolean }) {
  return (
    <button onClick={onClick} className="flex-1 rounded-[8px] px-3 py-2 mono text-[12px] transition-colors"
            style={{ border: `1px solid ${subtle ? "var(--paper-rule-2)" : "var(--accent)"}`, color: subtle ? "var(--ink-dim)" : "var(--accent)", background: "transparent" }}>
      {children}
    </button>
  );
}

function RoundBtn({ children, onClick, disabled }: { children: React.ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button onClick={onClick} disabled={disabled}
            className="w-11 h-11 rounded-full serif text-[22px] leading-none transition-colors flex items-center justify-center"
            style={{
              border: `1px solid ${disabled ? "var(--canvas-ink-faint)" : "var(--accent)"}`,
              color: disabled ? "var(--canvas-ink-faint)" : "var(--accent-soft)",
              opacity: disabled ? 0.4 : 1,
            }}>
      {children}
    </button>
  );
}

function Bars({ days, probs, top }: { days: string[]; probs: number[]; top: number }) {
  const max = Math.max(0.0001, ...probs);
  return (
    <div className="space-y-1.5">
      {days.map((d, i) => (
        <div key={d} className="flex items-center gap-2">
          <span className="mono text-[11px] w-9 text-right" style={{ color: i === top ? "var(--accent)" : "var(--ink-dim)" }}>{d.slice(0, 3)}</span>
          <div className="flex-1 h-[14px] rounded-[3px] overflow-hidden" style={{ background: "var(--paper-rule)" }}>
            <div className="h-full transition-[width] duration-150"
                 style={{ width: `${(probs[i] / max) * 100}%`, background: i === top ? "var(--accent)" : "var(--ink-faint)" }} />
          </div>
          <span className="mono text-[10px] w-8 text-right text-[var(--ink-faint)]">{(probs[i] * 100).toFixed(0)}</span>
        </div>
      ))}
    </div>
  );
}

function MethodsPanel({ K }: { K: SteerConstants }) {
  return (
    <details className="mt-12 rounded-[12px]" style={{ background: "var(--paper-2)", border: "1px solid var(--paper-rule)" }}>
      <summary className="cursor-pointer px-5 py-4 mono text-[12px] tracking-[0.12em] uppercase text-[var(--ink-2)] select-none">
        Methods &amp; validation
      </summary>
      <div className="px-5 pb-5 grid gap-6 md:grid-cols-2 text-[13px] text-[var(--ink-dim)] leading-[1.55]">
        <div>
          <h4 className="mono text-[11px] uppercase tracking-[0.12em] text-[var(--ink-faint)] mb-2">What runs here</h4>
          <p>
            GPT-2 small is split in two at <span className="mono">blocks.{K.layer}.hook_resid_pre</span> and
            exported to ONNX: an <em>encoder</em> (embeddings + blocks 0–{K.layer - 1}) and a
            <em> decoder</em> (block {K.layer} + final norm + unembed). Both run client-side via
            ONNX Runtime Web — the encoder once per prompt, the decoder on every steer. The
            rotation, projection and ablation are plain JS on the residual between the halves.
          </p>
        </div>
        <div>
          <h4 className="mono text-[11px] uppercase tracking-[0.12em] text-[var(--ink-faint)] mb-2">Why you can trust it</h4>
          <ul className="space-y-1.5">
            <li>· the fp32 split round-trips against TransformerLens to <span className="mono">cosine 0.99999</span>.</li>
            <li>· the deployed weights are int8-quantized (per-channel) to load on phones — they round-trip to <span className="mono">cosine ~0.997</span>, and the steering behavior is unchanged.</li>
            <li>· steer → prediction-shift tracks for <span className="mono">6/7</span> days (the Sat→Sun→Mon seam is the honest miss).</li>
            <li>· ablating the circle removes <span className="mono">~56%</span> of the successor probability; a matched random plane removes <span className="mono">~0%</span>.</li>
          </ul>
        </div>
        <div className="md:col-span-2">
          <h4 className="mono text-[11px] uppercase tracking-[0.12em] text-[var(--ink-faint)] mb-2">Honest scope</h4>
          <p>
            This reproduces and makes interactive a published finding (Engels et&nbsp;al., <em>Not All
            Language Model Features Are Linear</em>, 2024) — it is not a new result. The circle is
            geometrically present at every layer but only becomes causally load-bearing in the
            last couple; the steering effect is real but partial, and cleaner for days than months
            in GPT-2-small. The constants (layer, basis, day angles) were recovered offline and
            baked; the model math is all live.
          </p>
        </div>
      </div>
    </details>
  );
}
