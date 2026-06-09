import { decode } from "gpt-tokenizer/esm/encoding/r50k_base";
import type { SteerConstants, Prediction, OffPlane } from "./types";

/**
 * The in-plane rotation, matching the validated Phase-1/2 NumPy exactly.
 *
 * Decompose the final-position residual r about the circle center c0:
 *   a = (r-c0)·u1,  b = (r-c0)·u2,   offplane = (r-c0) - a·u1 - b·u2  (⊥ to the plane)
 * Rotate the in-plane part to target angle θ, keeping the example's own radius:
 *   a' = ρ·cosθ,  b' = ρ·sinθ,  with ρ = hypot(a,b)
 * Recombine, scaling the off-plane remainder by `offPlane`:
 *   r' = c0 + a'·u1 + b'·u2 + offPlane·offplane
 *   offPlane = 1 → preserve (r' = r + (a'-a)u1 + (b'-b)u2),  offPlane = 0 → isolate.
 *
 * Mutates a COPY of the row in `resid` at the final position; returns the new buffer.
 */
export function rotateResidual(
  resid: Float32Array,
  seq: number,
  dModel: number,
  k: SteerConstants,
  theta: number,
  offPlane: OffPlane,
): Float32Array {
  const out = resid.slice(); // copy; never mutate the pristine encoder output
  const base = (seq - 1) * dModel;
  const { u1, u2, center } = k;

  let a = 0, b = 0;
  for (let i = 0; i < dModel; i++) {
    const d = out[base + i] - center[i];
    a += d * u1[i];
    b += d * u2[i];
  }
  const rho = Math.hypot(a, b);
  const a2 = rho * Math.cos(theta);
  const b2 = rho * Math.sin(theta);

  if (offPlane === 1) {
    // preserve: add only the in-plane delta
    const da = a2 - a, db = b2 - b;
    for (let i = 0; i < dModel; i++) out[base + i] += da * u1[i] + db * u2[i];
  } else {
    // r' = c0 + a2·u1 + b2·u2 + offPlane·(r - c0 - a·u1 - b·u2)
    for (let i = 0; i < dModel; i++) {
      const offplane = out[base + i] - center[i] - a * u1[i] - b * u2[i];
      out[base + i] = center[i] + a2 * u1[i] + b2 * u2[i] + offPlane * offplane;
    }
  }
  return out;
}

/**
 * Necessity test: collapse a 2-D subspace at the final position by removing the
 * residual's in-plane component (relative to the circle center). `kind:"circle"`
 * removes the day-circle (prediction should degrade); `kind:"random"` removes a
 * fixed random plane of the same dimensionality (control — should barely move).
 * Verified offline: circle removes ~56% of successor prob, random ~0%.
 */
export function ablateResidual(
  resid: Float32Array,
  seq: number,
  dModel: number,
  k: SteerConstants,
  kind: "circle" | "random",
): Float32Array {
  const out = resid.slice();
  const base = (seq - 1) * dModel;
  const e1 = kind === "circle" ? k.u1 : k.random_u1;
  const e2 = kind === "circle" ? k.u2 : k.random_u2;
  let a = 0, b = 0;
  for (let i = 0; i < dModel; i++) {
    const d = out[base + i] - k.center[i];
    a += d * e1[i];
    b += d * e2[i];
  }
  for (let i = 0; i < dModel; i++) out[base + i] -= a * e1[i] + b * e2[i];
  return out;
}

/** Project the final-position residual onto the circle plane → (angle, radius). */
export function projectAngle(
  resid: Float32Array,
  seq: number,
  dModel: number,
  k: SteerConstants,
): { angle: number; radius: number } {
  const base = (seq - 1) * dModel;
  let a = 0, b = 0;
  for (let i = 0; i < dModel; i++) {
    const d = resid[base + i] - k.center[i];
    a += d * k.u1[i];
    b += d * k.u2[i];
  }
  return { angle: Math.atan2(b, a), radius: Math.hypot(a, b) };
}

/** Read off the day distribution + the true overall top token from logits. */
export function readPrediction(logits: Float32Array, k: SteerConstants): Prediction {
  // softmax over the 7 day tokens
  const dl = k.day_token_ids.map((id) => logits[id]);
  const mx = Math.max(...dl);
  const exps = dl.map((x) => Math.exp(x - mx));
  const sum = exps.reduce((s, x) => s + x, 0);
  const dayProbs = exps.map((x) => x / sum);
  let topDay = 0;
  for (let i = 1; i < dayProbs.length; i++) if (dayProbs[i] > dayProbs[topDay]) topDay = i;

  // overall argmax across the whole vocab (full softmax for the prob)
  let best = 0, bestVal = logits[0], gmax = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
    if (logits[i] > gmax) gmax = logits[i];
  }
  let z = 0;
  for (let i = 0; i < logits.length; i++) z += Math.exp(logits[i] - gmax);
  const overallProb = Math.exp(bestVal - gmax) / z;
  let overallTop: string;
  try { overallTop = decode([best]); } catch { overallTop = `<${best}>`; }

  return { dayProbs, topDay, overallTop, overallProb };
}
