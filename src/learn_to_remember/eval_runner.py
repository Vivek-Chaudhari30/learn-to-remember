"""Load a model, run eval conditions, write result files. Used by
scripts/run_eval.py (baselines, trained-policy eval) and scripts/probe_models.py.

Per (company, condition) it writes, under `out_dir`:
  <company>__<condition>.json          meta + summary (with CIs) + per-question rows
  <company>__<condition>_samples.json  a few readable trajectories per qtype
  rollouts/<company>__<condition>.jsonl  every trajectory (gitignored)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

from .env.episode import EnvConfig
from .env.tokens import HFTokenCounter
from .eval import ops_line, run_condition, sample_trajectories, summarize, trajectory_record

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPANIES = {"company_a": 0, "company_b": 1}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_policy(model_id: str, *, adapter: str | None = None, dtype: str = "auto",
                device: str = "auto"):
    import torch  # noqa: F401
    from .model_utils import get_device, load_model_and_tokenizer, load_model_with_adapter

    dev = get_device(device)
    if adapter:
        return load_model_with_adapter(model_id, adapter, dtype_name=dtype, device=dev)
    return load_model_and_tokenizer(model_id, dtype_name=dtype, device=dev)


def load_eval_questions(company: str, ws) -> list[dict]:
    """The regenerated eval split, checked against the committed JSONL (ids, gold,
    qtypes), so a different machine can't silently evaluate different questions."""
    path = os.path.join(ROOT, "data", "eval", f"{company}_eval.jsonl")
    with open(path) as f:
        committed = [json.loads(line) for line in f]
    gen = ws.splits["eval"]
    key = lambda q: (q["id"], q["gold"], json.dumps(q["qtype_by_version"], sort_keys=True))  # noqa: E731
    if [key(q) for q in gen] != [key(q) for q in committed]:
        raise RuntimeError(f"regenerated {company} eval set differs from {path}")
    return gen


def env_meta(model, tok, model_id, adapter, dtype_name) -> dict:
    import torch
    import transformers
    dev = next(model.parameters()).device
    meta = {"model_id": model_id, "adapter": adapter, "dtype": str(next(model.parameters()).dtype),
            "dtype_arg": dtype_name, "device": str(dev), "git_commit": git_commit(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "host": platform.node(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dev.type == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(dev)
    return meta


def run_jobs(*, model_id: str, companies, conditions, out_dir: str, subset: str = "all",
             limit: int | None = None, adapter: str | None = None, dtype: str = "auto",
             device: str = "auto", gen_batch: int = 16, episodes_per_rollout: int = 64,
             config: str = "configs/base.yaml", n_boot: int = 2000, greedy: bool = True,
             model_bundle=None, log=print) -> list[dict]:
    from .grpo.rollout import HFGenerator, chat_format_for
    from .model_utils import load_config
    from .workspace.generate import generate_company

    cfg_all = load_config(os.path.join(ROOT, config) if not os.path.isabs(config) else config)
    cfg = EnvConfig.from_dict(cfg_all["env"], cfg_all["reward"])
    model, tok, _ = model_bundle or load_policy(model_id, adapter=adapter, dtype=dtype,
                                                device=device)
    fmt = chat_format_for(model_id, tok)
    gen = HFGenerator(model, tok, stop_ids=fmt.stop_ids, max_new_tokens=cfg.max_gen_tokens,
                      greedy=greedy, batch_size=gen_batch)
    counter = HFTokenCounter(tok)
    meta = env_meta(model, tok, model_id, adapter, dtype)
    probe_ids = None
    if subset == "probe":
        with open(os.path.join(ROOT, "data", "eval", "probe_company_a.json")) as f:
            probe_ids = json.load(f)["ids"]
    os.makedirs(os.path.join(out_dir, "rollouts"), exist_ok=True)
    done = []
    for company in companies:
        ws = generate_company(COMPANIES[company], company, with_train=(company == "company_a"))
        qs = load_eval_questions(company, ws)
        if probe_ids is not None:
            if company != "company_a":
                raise ValueError("the probe subset is defined on company_a")
            keep = set(probe_ids)
            qs = [q for q in qs if q["id"] in keep]
        if limit:
            qs = qs[:limit]
        for cond in conditions:
            tag = f"{company}__{cond}"
            t0 = time.perf_counter()

            def progress(n, total, rows, _t0=t0, _tag=tag):
                acc = sum(r["correct"] for r in rows) / len(rows)
                el = time.perf_counter() - _t0
                log(f"  {_tag}: {n}/{total}  acc {acc:.3f}  {el / n:.2f} s/question  "
                    f"({el / 60:.1f} min)")

            log(f"{tag}: {len(qs)} questions, model {model_id}"
                + (f" + {adapter}" if adapter else ""))
            rows, trajs = run_condition(gen, fmt, ws, qs, cond, cfg, counter,
                                        episodes_per_rollout=episodes_per_rollout,
                                        progress=progress)
            wall = time.perf_counter() - t0
            summary = summarize(rows, n_boot=n_boot)
            record = {"meta": {**meta, "company": company, "condition": cond, "subset": subset,
                               "limit": limit, "n_questions": len(qs), "greedy": greedy,
                               "gen_batch": gen_batch, "episodes_per_rollout": episodes_per_rollout,
                               "wall_seconds": wall, "seconds_per_question": wall / len(qs),
                               "env": cfg.__dict__},
                      "summary": summary, "rows": rows}
            with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
                json.dump(record, f, indent=1, sort_keys=True)
            with open(os.path.join(out_dir, f"{tag}_samples.json"), "w") as f:
                json.dump(sample_trajectories(rows, trajs, qs), f, indent=1)
            qby = {q["id"]: q for q in qs}
            with open(os.path.join(out_dir, "rollouts", f"{tag}.jsonl"), "w") as f:
                for r, t in zip(rows, trajs):
                    f.write(json.dumps(trajectory_record(t, qby[r["id"]])) + "\n")
            log(f"{tag}: accuracy {summary['overall']['accuracy']['value']:.3f}, "
                f"wall {wall / 60:.1f} min ({wall / len(qs):.2f} s/question)")
            log("  " + ops_line(summary))
            done.append(record)
            sys.stdout.flush()
    return done
