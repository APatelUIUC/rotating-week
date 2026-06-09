import * as ort from "onnxruntime-web";

// Load ORT's wasm + worker glue from the CDN matching our installed version.
// Vite's dev server won't transform the .mjs worker loader if it lives in
// /public, so a cross-origin CDN is the reliable path in both dev and build;
// COEP:credentialless (see vite.config) keeps us cross-origin-isolated so the
// threaded SharedArrayBuffer build still works.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 8);

// Models are hosted on the HF Hub (free egress, CDN-backed) and fetched at
// runtime — they're far too large to ship in the Vercel deploy. HF reflects the
// request Origin into Access-Control-Allow-Origin, so the cross-origin fetch
// works under our COEP:credentialless isolation.
const HF_BASE = "https://huggingface.co/lapiskasha/gpt2-circular-steering/resolve/main";
// int8 per-channel quantized (~169MB total vs 655MB fp32). Round-trips to cosine
// ~0.997 and the steering lockstep is unchanged (5.6/7) — but the ~4x smaller
// footprint is what lets it load on phones without OOM-crashing mobile WebKit.
const PRE_URL = `${HF_BASE}/model_pre_q.onnx`;
const POST_URL = `${HF_BASE}/model_post_q.onnx`;

export interface LoadProgress {
  /** 0..1 over both model downloads combined. */
  frac: number;
  receivedMB: number;
  totalMB: number;
  stage: string;
}

let pre: ort.InferenceSession | null = null;
let post: ort.InferenceSession | null = null;

async function fetchWithProgress(
  url: string,
  onChunk: (received: number, total: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onChunk(received, total);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

/** Download + instantiate both halves. Idempotent. */
export async function initSteering(onProgress: (p: LoadProgress) => void): Promise<void> {
  if (pre && post) return;
  const PRE_BYTES = 122 * 1e6, POST_BYTES = 47 * 1e6, TOTAL = PRE_BYTES + POST_BYTES;
  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"] };
  let done = 0;
  const report = (recv: number, stage: string) =>
    onProgress({
      frac: Math.min(1, (done + recv) / TOTAL),
      receivedMB: Math.round((done + recv) / 1e6),
      totalMB: Math.round(TOTAL / 1e6),
      stage,
    });

  // Load each half sequentially and build its session before fetching the next,
  // so the big ArrayBuffer becomes garbage-collectable immediately. Peak memory
  // stays near a single model (~120MB) rather than both at once — this is what
  // keeps it from OOM-crashing on mobile WebKit (iOS Safari / iOS Chrome).
  pre = await ort.InferenceSession.create(
    await fetchWithProgress(PRE_URL, (r) => report(r, "Downloading model (1 / 2)")), opts);
  done = PRE_BYTES;
  post = await ort.InferenceSession.create(
    await fetchWithProgress(POST_URL, (r) => report(r, "Downloading model (2 / 2)")), opts);
  onProgress({ frac: 1, receivedMB: Math.round(TOTAL / 1e6), totalMB: Math.round(TOTAL / 1e6), stage: "Starting up" });
}

/** Run the encoder once for a prompt → residual stream at blocks.L.hook_resid_pre. */
export async function runPre(inputIds: number[]): Promise<{ resid: Float32Array; seq: number; dModel: number }> {
  if (!pre) throw new Error("not initialized");
  const seq = inputIds.length;
  const t = new ort.Tensor("int64", BigInt64Array.from(inputIds.map((i) => BigInt(i))), [1, seq]);
  const out = await pre.run({ input_ids: t });
  const resid = out["resid"];
  const data = new Float32Array(resid.data as Float32Array);
  const dModel = (resid.dims[2] as number) ?? data.length / seq;
  return { resid: data, seq, dModel };
}

/** Run the decoder on a (possibly intervened) residual → final-position logits. */
export async function runPost(resid: Float32Array, seq: number, dModel: number): Promise<Float32Array> {
  if (!post) throw new Error("not initialized");
  const t = new ort.Tensor("float32", resid, [1, seq, dModel]);
  const out = await post.run({ resid: t });
  const logits = out["logits"];
  const data = logits.data as Float32Array;
  const vocab = logits.dims[2] as number;
  // last position only
  return new Float32Array(data.subarray((seq - 1) * vocab, seq * vocab));
}

export function isReady(): boolean { return !!(pre && post); }
