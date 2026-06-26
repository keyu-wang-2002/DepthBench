"""
Route tokenized shards into train/eval/test by a seeded draw. Pure relabeling:
symlinks shards from the flat pool into split dirs, records the exact assignment
and per-split token budgets. Re-runnable; change --seed/--n-* and rerun freely.

Example usage:
```bash
python split_tokenized_data.py \
    --data-dir data/fineweb-edu/pre-tokenize/train \
    --out-dir  data/fineweb-edu/splits \
    --n-val 3 \
    --n-test 0 \
    --seed 42
```

This script generates up to 3 folders at `out_dir`: train/, val/, test/, each containing 
only the symlinks to the original shards in `data_dir`. 
It also generates a `split_manifest.json` file in `out_dir` that records the split configuration 
and arguments to rerun this script.
"""

import argparse
import json
import random
import pathlib


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)   # e.g. data/fineweb-edu-350BT/tokenized/
    p.add_argument("--out-dir",  required=True)   # e.g. data/fineweb-edu-350BT/tokenized_splits/
    p.add_argument("--n-val",  type=int, default=3, help="number of shards to reserve for eval")
    p.add_argument("--n-test", type=int, default=0, help="number of shards to reserve for test")
    p.add_argument("--seed",   type=int, default=42, help="for reproduciblte shuffling of shards")
    args = p.parse_args()

    data_dir = pathlib.Path(args.data_dir).resolve()
    shards = sorted(data_dir.glob("*.npy"))
    if not shards:
        raise SystemExit(f"no .npy in {data_dir}")

    order = shards[:]
    random.Random(args.seed).shuffle(order)
    test  = order[:args.n_test]
    val   = order[args.n_test:args.n_test + args.n_val]
    train = order[args.n_test + args.n_val:]

    def toks(npy):
        m = npy.with_suffix(".meta.json")
        return json.loads(m.read_text())["num_tokens"] if m.exists() else 0

    out = pathlib.Path(args.out_dir)
    manifest = {
        "seed": args.seed, 
        "data_dir": str(data_dir), 
        "n_test": args.n_test, 
        "n_val": args.n_val, 
        "splits": {}
    }
    for name, group in [("train", train), ("eval", val), ("test", test)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):                      # clear previous run
            if old.is_symlink():
                old.unlink()
        total = 0
        for npy in group:
            for f in data_dir.glob(npy.stem + ".*"):      # .npy + .meta.json + .csv.gz
                (d / f.name).symlink_to(f)
            total += toks(npy)
        manifest["splits"][name] = {
            "num_shards": len(group), 
            "num_tokens": total,
            "shards": [s.name for s in group]
        }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    for k, v in manifest["splits"].items():
        print(f"{k:5s} {v['num_shards']:>4} shards  {v['num_tokens']:>15,} tokens")
# end of file