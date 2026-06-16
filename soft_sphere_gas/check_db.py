"""Sanity-check a soft-sphere sweep DB against the analytic bounds (paper units).

For every completed run it verifies the physics that *must* hold for a converged variational
calculation, reading the energies straight from the DB (``e_per_n`` is already stored in
per-particle paper units — no progress-bar guesswork):

* **lower bound** : E/N >= 4πx           (Lieb-Yngvason; rigorous)
* **upper bound** : E/N <= Eq.31          (first-order UB; variational)
* **Lee-Yang**    : E/N ~ Lee-Yang(x)     (the x->0 universal limit; we expect slightly above)

It also recomputes 4πx / Lee-Yang / Eq.31 independently from ``dilute_gas`` and cross-checks the
Eq.31 value stored in the DB, prints the queue status and any failed runs, and shows the
seed-aggregated curve (what ``load_curve`` feeds the plot).

    python check_db.py                       # default outputs/soft_sphere.db
    python check_db.py --db ~/dilute-bose/outputs/soft_sphere.db
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import db
from dilute_gas import first_order_energy_upper_bound as eq31
from dilute_gas import lee_yang_energy_per_particle as lee_yang
from sweep import load_curve

# colour only on a real terminal, so redirected/piped output stays plain
if sys.stdout.isatty():
    GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
else:
    GREEN = RED = YELLOW = DIM = RESET = ""


def _v(row_verdict_json: str, key, default=None):
    try:
        return json.loads(row_verdict_json).get(key, default)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="outputs/soft_sphere.db")
    ap.add_argument("--R", type=float, default=10.0, help="potential R (for Eq.31 recompute); SS10=10")
    args = ap.parse_args()

    conn = db.connect(args.db)
    print(f"=== {args.db} ===")
    print(f"queue status: {db.status_counts(conn)}\n")

    rows = conn.execute(
        "SELECT * FROM runs WHERE status='done' ORDER BY x, N, seed"
    ).fetchall()
    if not rows:
        print("no completed runs yet.")
        _print_failures(conn)
        return

    print("per-run check (paper units, a=1):")
    hdr = (f"{'x':>10} {'N':>4} {'seed':>4} {'E/N':>11} {'err':>9} "
           f"{'4pix':>10} {'LeeYang':>10} {'Eq31':>10} {'E/LY':>6} {'corridor':>9} "
           f"{'verdict':>7} {'epochs':>7}")
    print(hdr)
    print("-" * len(hdr))

    n_viol = 0
    for r in rows:
        x, N = r["x"], r["N"]
        e, err = r["e_per_n"], r["err_per_n"] or 0.0
        lb = 4 * math.pi * x
        ly = lee_yang(x)
        ub = eq31(x, _v0_for_R(args.R), args.R)

        # cross-check the Eq.31 stored at run time against a fresh recompute
        ub_db = r["upper_bound"]
        ub_mismatch = ub_db is not None and abs(ub_db - ub) > 1e-9 * max(abs(ub), 1e-30)

        below_lb = e < lb - err            # clear lower-bound violation (beyond the error bar)
        above_ub = e > ub + err            # clear upper-bound (variational) violation
        ok = not (below_lb or above_ub)
        if not ok:
            n_viol += 1
        tag = f"{GREEN}ok{RESET}" if ok else f"{RED}VIOLATION{RESET}"
        if ub_mismatch:
            tag += f" {RED}[Eq31!]{RESET}"

        epochs = _v(r["verdict_json"], "epochs_ran", "?")
        reason = _v(r["verdict_json"], "early_stop_reason")
        ep_str = f"{epochs}{('/' + reason) if reason else ''}"
        verdict = f"{GREEN}PASS{RESET}" if r["passed"] else f"{YELLOW}fail{RESET}"

        print(f"{x:>10.3e} {N:>4} {r['seed']:>4} {e:>11.4e} {err:>9.2e} "
              f"{lb:>10.4e} {ly:>10.4e} {ub:>10.4e} {e/ly:>6.2f} {tag:>9} "
              f"{verdict:>7} {ep_str:>7}")

    print()
    if n_viol == 0:
        print(f"{GREEN}all {len(rows)} done points are inside [4pix, Eq31] (physics OK){RESET}")
    else:
        print(f"{RED}{n_viol}/{len(rows)} points VIOLATE the corridor — investigate{RESET}")
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"{DIM}verdict passed: {n_pass}/{len(rows)} (fail can just mean not-yet-stationary){RESET}")

    _print_failures(conn)
    _print_curve(conn, rows)


def _v0_for_R(R: float) -> float:
    from point import Potential
    return Potential.from_R(R).V0_paper


def _print_failures(conn):
    fails = conn.execute(
        "SELECT potential_label, x, N, seed, error FROM runs WHERE status='failed'"
    ).fetchall()
    if fails:
        print(f"\n{RED}failed runs:{RESET}")
        for f in fails:
            line1 = (f["error"] or "").strip().splitlines()[-1:] or [""]
            print(f"  {f['potential_label']} x={f['x']:g} N={f['N']} seed={f['seed']}: {line1[0]}")


def _print_curve(conn, rows):
    labels_Ns = sorted({(r["potential_label"], r["N"]) for r in rows})
    for label, N in labels_Ns:
        try:
            c = load_curve(conn, label, N)
        except Exception as exc:
            print(f"\n{label} N={N}: load_curve failed: {exc!r}")
            continue
        if len(c["x"]) == 0:
            continue
        print(f"\nseed-aggregated curve ({label}, N={N}, verdict-passing): "
              f"{len(c['x'])} x-points (seeds/x: {c['n_per_x']})")
        for x, e, err in zip(c["x"], c["e_per_n"], c["err"]):
            print(f"  x={x:.3e}  E/N={e:.5e} +/- {err:.2e}   E/N / 4pix = {e/(4*math.pi*x):.3f}")


if __name__ == "__main__":
    main()
