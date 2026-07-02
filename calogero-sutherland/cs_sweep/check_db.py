"""Sanity-check a CS sweep DB: queue status + every done point vs the exact E0 = N(1+L(N-1)).

    python check_db.py --db outputs/runq.db

Each done row prints E/N, the exact E/N, the variational gap (must be ≥ 0 up to MC error — the
variational principle), the verdict, and epochs run. A negative gap beyond a few error bars means a
bias (e.g. the epsilon-softened potential, or a bug), not a better-than-exact result.
"""

from __future__ import annotations

import argparse
import json

from runq import store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=store.DEFAULT_DB)
    args = ap.parse_args()
    conn = store.connect(args.db)

    print(f"status: {store.status_counts(conn)}\n")

    rows = store.fetch(conn, "done")
    if not rows:
        print("no done rows yet.")
    else:
        hdr = f"{'L':>6} {'N':>4} {'seed':>4} {'kind':>12} {'E/N':>10} {'exact/N':>10} {'gap':>9} {'pass':>4} {'epochs':>6}"
        print(hdr)
        print("-" * len(hdr))
        n_bad = 0
        for r in rows:
            p = json.loads(r["params_json"])
            res = json.loads(r["result_json"])
            N = p["N"]
            gap = res.get("gap", float("nan"))
            err = res.get("err_total") or 0.0
            # variational gap should be >= 0 within a few sigma
            bad = gap < -3 * err - 1e-9
            n_bad += bad
            flag = "  <-- below exact!" if bad else ""
            print(f"{p['L']:>6g} {N:>4} {p['seed']:>4} {p['kind']:>12} "
                  f"{res['e_per_n']:>10.4f} {res['e_exact']/N:>10.4f} {gap:>9.4f} "
                  f"{('Y' if res['passed'] else 'n'):>4} {res.get('epochs_ran','?'):>6}{flag}")
        print(f"\n{len(rows)} done; {n_bad} below exact beyond 3σ "
              f"({'all variational ✓' if n_bad == 0 else 'investigate ⚠'}).")

    failed = store.fetch(conn, "failed")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for r in failed:
            p = json.loads(r["params_json"])
            tail = (r["error"] or "").strip().splitlines()[-1:] or [""]
            print(f"  L{p['L']:g}_N{p['N']} seed={p['seed']}: {tail[0]}")


if __name__ == "__main__":
    main()
