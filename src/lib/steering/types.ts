/** Frontend constants baked by Phase 2 (circular-steering/phase2/constants.json). */
export interface SteerConstants {
  layer: number;
  d_model: number;
  days: string[];
  day_token_ids: number[];
  u1: number[];
  u2: number[];
  center: number[];
  day_angles_rad: number[];
  /** a fixed random orthonormal 2-D plane (seed 0) — the control for the
   *  ablation/necessity test. */
  random_u1: number[];
  random_u2: number[];
  frame: string;
}

export interface PromptDef {
  text: string;
  input_ids: number[];
  last_day: number;
  baseline_pred: number;
  isolate_score: number;
}

/** Off-plane retention: 0 = isolate (drop competing off-circle signal, crisp
 *  lockstep), 1 = preserve (the brief's canonical "rotate in-plane, keep
 *  everything"), values >1 amplify. The "mess around" dial. */
export type OffPlane = number;

export interface Prediction {
  /** softmax over the 7 day tokens only (for the bars). */
  dayProbs: number[];
  /** index of the argmax day. */
  topDay: number;
  /** the single highest-logit token in the whole vocab, decoded (honesty cue). */
  overallTop: string;
  /** probability mass of overallTop. */
  overallProb: number;
}
