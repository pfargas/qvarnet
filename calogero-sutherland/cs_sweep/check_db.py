"""Sanity-check a CS sweep DB: queue status + every done point vs the exact E0 = N(1+L(N-1)).

    python check_db.py --db outputs/cs.db

Each done row prints E/N, the exact E/N, the variational gap (must be ≥ 0 up to MC error — the
variational principle), the verdict, and epochs run. A negative gap beyond a few error bars means a
bias (e.g. the epsilon-softened potential, or a bug), not a better-than-exact result.
"""

from __future__ import annotations

import argparse
import json

import db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=db.DEFAULT_DB)
    args = ap.parse_args()
    conn = db.connect(args.db)

    print(f"status: {db.status_counts(conn)}\n")

    rows = db.fetch_done(conn)
    if not rows:
        print("no done rows yet.")
    else:
        hdr = f"{'L':>6} {'N':>4} {'seed':>4} {'kind':>12} {'E/N':>10} {'exact/N':>10} {'gap':>9} {'pass':>4} {'epochs':>6}"
        print(hdr)
        print("-" * len(hdr))
        n_bad = 0
        for r in rows:
            phys = json.loads(r["physics_json"])
            hp = json.loads(r["hp_json"])
            N = phys["N"]
            gap = r["gap"] if r["gap"] is not None else float("nan")
            err = r["err_total"] or 0.0
            # variational gap should be >= 0 within a few sigma
            bad = gap < -3 * err - 1e-9
            n_bad += bad
            flag = "  <-- below exact!" if bad else ""
            print(f"{phys['L']:>6g} {N:>4} {r['seed']:>4} {hp['kind']:>12} "
                  f"{r['e_per_n']:>10.4f} {r['e_exact']/N:>10.4f} {gap:>9.4f} "
                  f"{('Y' if r['passed'] else 'n'):>4} {json.loads(r['verdict_json']).get('epochs_ran','?'):>6}{flag}")
        print(f"\n{len(rows)} done; {n_bad} below exact beyond 3σ "
              f"({'all variational ✓' if n_bad == 0 else 'investigate ⚠'}).")

    failed = conn.execute("SELECT physics_json, seed, error FROM runs WHERE status='failed'").fetchall()
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for r in failed:
            phys = json.loads(r["physics_json"])
            tail = (r["error"] or "").strip().splitlines()[-1:] or [""]
            print(f"  L{phys['L']:g}_N{phys['N']} seed={r['seed']}: {tail[0]}")


if __name__ == "__main__":
    main()
