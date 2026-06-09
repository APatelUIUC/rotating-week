import { useCallback, useEffect, useRef, useState } from "react";
import { encode } from "gpt-tokenizer/esm/encoding/r50k_base";
import type { SteerConstants, PromptDef, Prediction } from "../lib/steering/types";
import { initSteering, runPre, runPost, type LoadProgress } from "../lib/steering/onnx";
import { rotateResidual, ablateResidual, projectAngle, readPrediction } from "../lib/steering/intervention";
import DayCircle from "./DayCircle";

type Active = { text: string; input_ids: number[] };
const GPT2_BOS = 50256;

const PAPER_URL = "https://arxiv.org/abs/2405.14860";
const CODE_URL = "https://github.com/APatelUIUC/rotating-week";
const AUTHOR_URL = "https://www.akashpa.tel";

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
  const [theta, setTheta] = useState(0);
  const [offPlane, setOffPlane] = useState(0);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [baselinePred, setBaselinePred] = useState<Prediction | null>(null);
  const [baselineAngle, setBaselineAngle] = useState(0);
  const [ablation, setAblation] = useState<{ kind: "circle" | "random"; pred: Prediction } | null>(null);

  const residRef = useRef<{ resid: Float32Array; seq: number; dModel: number } | null>(null);
  const job = useRef<{ running: boolean; pending: { theta: number; off: number } | null }>({ running: false, pending: null });

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

  const decode = useCallback(async (th: number, off: number) => {
    if (!K_ || !residRef.current) return;
    if (job.current.running) { job.current.pending = { theta: th, off }; return; }
    job.current.running = true;
    try {
      const { resid, seq, dModel } = residRef.current;
      const r = rotateResidual(resid, seq, dModel, K_, th, off);
      setPred(readPrediction(await runPost(r, seq, dModel), K_));
    } finally {
      job.current.running = false;
      const p = job.current.pending;
      if (p) { job.current.pending = null; decode(p.theta, p.off); }
    }
  }, [K_]);

  useEffect(() => { if (ready && prompts.length && !active) setActive(prompts[0]); }, [ready, prompts, active]);

  useEffect(() => {
    if (!K_ || !ready || !active) return;
    let alive = true;
    (async () => {
      const enc = await runPre(active.input_ids);
      if (!alive) return;
      residRef.current = enc;
      const { angle } = projectAngle(enc.resid, enc.seq, enc.dModel, K_);
      setBaselineAngle(angle);
      setTheta(angle);
      setAblation(null);
      setBaselinePred(readPrediction(await runPost(enc.resid, enc.seq, enc.dModel), K_));
      decode(angle, offPlane);
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, ready]);

  const onTheta = (th: number) => { setTheta(th); setAblation(null); decode(th, offPlane); };
  const onOff = (off: number) => { setOffPlane(off); decode(theta, off); };
  const stepDay = (dir: 1 | -1) => {
    if (!K_) return;
    const cur = nearestDay(theta, K_.day_angles_rad);
    onTheta(K_.day_angles_rad[(cur + dir + K_.days.length) % K_.days.length]);
  };
  const runAblation = async (kind: "circle" | "random") => {
    if (!K_ || !residRef.current) return;
    const { resid, seq, dModel } = residRef.current;
    const r = ablateResidual(resid, seq, dModel, K_, kind);
    setAblation({ kind, pred: readPrediction(await runPost(r, seq, dModel), K_) });
  };
  const submitCustom = () => {
    const t = customText.trim();
    if (!t) return;
    setActive({ text: t, input_ids: [GPT2_BOS, ...encode(t)].slice(0, 64) });
  };

  if (err) return <Shell><div className="mono text-[var(--accent)] p-6">{err}</div></Shell>;
  if (!K_ || !ready || !active) return <Shell><Loader load={load} /></Shell>;

  const dialDay = nearestDay(theta, K_.day_angles_rad);
  const predDay = pred?.topDay ?? 0;
  const selectedCurated = prompts.findIndex((p) => p.text === active.text);
  const refDay = baselinePred?.topDay ?? predDay;
  const overallIsDay = pred ? K_.days.some((d) => pred.overallTop.trim() === d) : true;

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
            stream, and predicts the next day by <em>rotating around it</em>. Turn the dial:
            you rotate the model's internal state, and its next-token guess walks around the
            week in lockstep — one step ahead of wherever you point.
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

            <div className="flex justify-center py-2">
              <DayCircle
                days={K_.days}
                angles={K_.day_angles_rad}
                theta={theta}
                baselineAngle={baselineAngle}
                topDay={predDay}
                dayProbs={pred?.dayProbs ?? K_.days.map(() => 0)}
                onTheta={onTheta}
              />
            </div>

            <div className="flex items-center justify-center gap-5 mt-2 mb-1">
              <Readout label="you point at" value={K_.days[dialDay]} />
              <span className="text-[var(--accent)] text-2xl leading-none">→</span>
              <Readout label="model predicts" value={K_.days[predDay]} accent />
            </div>
            <p className="text-center mono text-[11px] text-[var(--canvas-ink-faint)] mt-3">
              drag the coral handle · or step a whole day below
            </p>
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
                    custom prompt — the rotation still acts on the final token, but the
                    lockstep is cleanest on day-sequence prompts.
                  </p>
                )}
              </div>
            </Field>

            <Field label="Step the dial">
              <div className="flex gap-2">
                <Btn onClick={() => stepDay(-1)}>− one day</Btn>
                <Btn onClick={() => stepDay(1)}>+ one day</Btn>
                <Btn onClick={() => onTheta(baselineAngle)} subtle>reset</Btn>
              </div>
            </Field>

            <Field label="Predicted next day">
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
                The rotation only ever touches the 2-D day-circle. This sets how much of the
                residual's <em>off-circle</em> content rides along: <strong>isolate&nbsp;(0)</strong> drops the
                competing signal for a crisp result; <strong>preserve&nbsp;(1)</strong> is the untouched
                residual. Push past 1 to amplify and watch it break.
              </p>
            </Field>

            <Field label="Is the circle necessary?">
              <p className="text-[12px] text-[var(--ink-dim)] leading-snug mb-3">
                Rotating shows the circle can <em>steer</em> the answer (sufficiency). This removes
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
        {load?.stage ?? "Fetching constants"} · {load?.receivedMB ?? 0}/{load?.totalMB ?? 655} MB
      </p>
      <div className="h-[3px] rounded-full overflow-hidden" style={{ background: "var(--paper-rule-2)" }}>
        <div className="h-full transition-[width] duration-200" style={{ width: `${pct}%`, background: "var(--accent)" }} />
      </div>
      <p className="mono text-[11px] text-[var(--ink-faint)] mt-3">{pct}% · one-time download, then cached by your browser</p>
    </div>
  );
}

function Readout({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="text-center">
      <div className="mono text-[10px] tracking-[0.14em] uppercase text-[var(--canvas-ink-faint)] mb-1">{label}</div>
      <div className="serif text-[24px] leading-none" style={{ color: accent ? "var(--accent-soft)" : "var(--canvas-ink)" }}>{value}</div>
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
            ONNX Runtime Web — the encoder once per prompt, the decoder on every dial-turn. The
            rotation, projection and ablation are plain JS on the residual between the halves.
          </p>
        </div>
        <div>
          <h4 className="mono text-[11px] uppercase tracking-[0.12em] text-[var(--ink-faint)] mb-2">Why you can trust it</h4>
          <ul className="space-y-1.5">
            <li>· split round-trips against TransformerLens to <span className="mono">cosine 0.99999</span> (max-abs-diff ~3e-5).</li>
            <li>· the in-browser rotation matches the TransformerLens patched result to the same tolerance.</li>
            <li>· rotation → succession lockstep holds <span className="mono">6/7</span> days (Sun→Mon is the honest miss).</li>
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
