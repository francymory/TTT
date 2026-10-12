"""Step-0 diagnostic for the TTT-NTP training-time write (Qwen3 TTT model, CPU, no gradients).

For each document and each (inner_opt, ttt_lr) setting it compares the next-token loss with TTT off vs on
(per 1024-token chunk; chunk 0 receives no write) and reports, per TTT layer,
  raw = ||lr * sum_c dW_c|| / ||W_down||                 (accumulated write, before norm_preserve)
  eff = ||W_eff_last - W_down|| / ||W_down||             (actual change of the last chunk's weight)
A setting is "safe" at step 0 if loss(on) - loss(off) on chunks >= 1 is small (e.g. < 0.1 nats).
This is a necessary condition, not a criterion for the best setting: that requires training + evaluation.

Run it on a compute node: login nodes kill processes after 600 s of CPU time (ulimit -t).
Example (GPU, ~minutes):
  srun -A IscrB_MOSAIC -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 00:30:00 \
  python -u scripts/ttt_write_diag.py --data /path/to/ldc/data --key text --ndoc 2 \
      --inner_opt sgd specnorm --ttt_lr 2.6 0.1 0.01 0.001 --out diag_scan.json > diag_scan.log 2>&1 &
"""
import argparse, gc, glob, json, os, resource, sys, traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hf_models  # noqa: F401
from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3MLP
from opt_einsum import contract
from transformers import AutoTokenizer

BASE = "/leonardo_work/IscrB_MOSAIC/franci/huggingface/models/Qwen3-0.6B-Base"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=BASE)
ap.add_argument("--seq", type=int, default=4096)
ap.add_argument("--ndoc", type=int, default=2)
ap.add_argument("--data", default="/leonardo_work/IscrB_MOSAIC/fwp/prolong/arxiv.jsonl",
                help=".jsonl file or directory of .parquet files")
ap.add_argument("--key", default="content_split", help="text field (ProLong: content_split, LDC: text)")
ap.add_argument("--inner_opt", nargs="+", default=["sgd"], choices=["sgd", "specnorm"])
ap.add_argument("--ttt_lr", nargs="+", type=float, default=[2.6])
ap.add_argument("--no_norm_preserve", action="store_true")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default=None, help="optional JSON file with all results")
args = ap.parse_args()
torch.set_num_threads(args.threads)


def log(*a):
    print(*a, flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---- model: release 0.6B foundation; inner_opt / lr are changed per run on the MLP modules ----
F = dict(ttt_layers=[0, 4, 8, 12, 16, 20, 24], ttt_mode=True, ttt_proj=True, ttt_proj_init="diag_shared",
         ttt_conv=False, ttt_predict_mode="next", ttt_target="hidden_states", ttt_chunk=1024,
         ttt_norm_preserve=not args.no_norm_preserve, ttt_inner_opt="sgd", ttt_lr=2.6)
cfg = Qwen3Config.from_pretrained(args.ckpt)
for k, v in F.items():
    setattr(cfg, k, v)
cfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM.from_pretrained(args.ckpt, config=cfg, dtype=torch.float32).to(args.device).eval()
if os.path.realpath(args.ckpt) == os.path.realpath(BASE):
    for i in F["ttt_layers"]:  # release diag_shared init, as the trainer does for the missing keys
        model._init_weights(model.model.layers[i].mlp.ttt_proj)
tok = AutoTokenizer.from_pretrained(args.ckpt)
ttt_mlps = [l.mlp for l in model.model.layers if getattr(l.mlp, "is_ttt", False)]

# ---- hook: measure the write of each TTT layer (same math as Qwen3MLP.forward) ----
stats = {}
_orig = Qwen3MLP.forward


def _fwd(self, x, t=None):
    out = _orig(self, x, t)
    if t is not None and getattr(self, "is_ttt", False):
        tp = self.padding(t)
        hp = self.padding(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        d = contract("btch,btcd,de->bteh", hp[:, :-1, :-1], tp[:, :-1, 1:], self.ttt_proj.weight)[0]
        if self.ttt_inner_opt == "specnorm":
            d = d / torch.linalg.matrix_norm(d, ord=2, keepdim=True).clamp_min(1e-6)
        W = self.down_proj.weight
        cum = self.ttt_lr * d.sum(0)
        W_eff = W + cum
        if self.ttt_norm_preserve:
            W_eff = W_eff / W_eff.norm(dim=-1, keepdim=True).clamp_min(1e-5) * W.norm(dim=-1, keepdim=True)
        stats[self.layer_idx] = {"raw": (cum.norm() / W.norm()).item(),
                                 "eff": ((W_eff - W).norm() / W.norm()).item()}
        del tp, hp, d, cum, W_eff
    return out


Qwen3MLP.forward = _fwd


# ---- data ----
def texts():
    if os.path.isdir(args.data):
        import pyarrow.parquet as pq
        for fn in sorted(glob.glob(os.path.join(args.data, "*.parquet"))):
            for batch in pq.ParquetFile(fn).iter_batches(batch_size=64, columns=[args.key]):
                yield from batch.column(0).to_pylist()
    else:
        with open(args.data) as f:
            for line in f:
                yield json.loads(line)[args.key]


docs = []
for text in texts():
    ids = tok(text[: 20 * args.seq])["input_ids"]  # enough characters for SEQ tokens, avoids huge tokenizations
    if len(ids) >= args.seq:
        docs.append(torch.tensor([ids[: args.seq]], device=args.device))
    if len(docs) == args.ndoc:
        break
log(f"device {args.device}  ckpt {args.ckpt}  seq {args.seq}  docs {len(docs)}/{args.ndoc}  data {args.data}")
if len(docs) < args.ndoc:
    log(f"WARNING: only {len(docs)} documents with >= {args.seq} tokens found")


@torch.inference_mode()
def loss(ids, ttt):
    for l in model.model.layers:
        l.is_ttt_layer = ttt and l.mlp.layer_idx in F["ttt_layers"]
    hid = model.model(input_ids=ids).last_hidden_state[0, :-1]
    nll = torch.cat([torch.nn.functional.cross_entropy(model.lm_head(hid[i:i + 512]).float(),
                                                       ids[0, 1 + i:513 + i], reduction="none")
                     for i in range(0, hid.shape[0], 512)])
    del hid
    return [nll[i:i + 1024].mean().item() for i in range(0, len(nll), 1024)]


# ---- scan ----
results = []
try:
    for di, ids in enumerate(docs):
        off = loss(ids, False)
        log(f"\ndoc{di}  TTT off per-chunk {[round(v, 2) for v in off]}   (max RSS {rss_gb():.1f} GB)")
        for opt in args.inner_opt:
            for lr in args.ttt_lr:
                for m in ttt_mlps:
                    m.ttt_inner_opt, m.ttt_lr = opt, lr
                stats.clear()
                on = loss(ids, True)
                dl = sum(a - b for a, b in zip(on[1:], off[1:])) / max(len(on) - 1, 1)
                r = {"doc": di, "inner_opt": opt, "ttt_lr": lr, "off": off, "on": on,
                     "dloss_chunks_ge1": dl, "layers": dict(sorted(stats.items()))}
                results.append(r)
                log(f"doc{di}  {opt:8s} lr={lr:<8g} dloss(chunks>=1)={dl:+7.3f}  on {[round(v, 2) for v in on]}")
                log("         raw " + " ".join(f"L{k}:{v['raw']:.2g}" for k, v in r["layers"].items()))
                log("         eff " + " ".join(f"L{k}:{v['eff']:.2g}" for k, v in r["layers"].items()))
                gc.collect()
        gc.collect()
except Exception:
    log("ERROR during scan:\n" + traceback.format_exc())

# ---- summary: mean over documents ----
log("\n===== summary (mean over docs) =====")
log(f"{'inner_opt':9s} {'ttt_lr':>8s} {'dloss':>8s} {'max eff':>8s} {'max raw':>9s}  verdict (dloss<0.1)")
for opt in args.inner_opt:
    for lr in args.ttt_lr:
        rs = [r for r in results if r["inner_opt"] == opt and r["ttt_lr"] == lr]
        if not rs:
            continue
        dl = sum(r["dloss_chunks_ge1"] for r in rs) / len(rs)
        me = max(v["eff"] for r in rs for v in r["layers"].values())
        mr = max(v["raw"] for r in rs for v in r["layers"].values())
        log(f"{opt:9s} {lr:8g} {dl:+8.3f} {me:8.3g} {mr:9.3g}  {'safe' if abs(dl) < 0.1 else 'BREAKS'}  (n_docs={len(rs)})")
log(f"max RSS {rss_gb():.1f} GB")
if args.out:
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    log(f"wrote {args.out}")
