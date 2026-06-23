#!/usr/bin/env python3
"""Compare necessity (nec_hdel) under different LOCALIZATION sources on the
SAME images (positional pairing; runs share seed-123 image order).

Inputs: xmodel_{attnloc,friloc,inflowloc}_g50.json (each has per-model
nec_hdel_arr / loc_hdel_arr). Asks: does FRI-localization match/beat the
attn (weak) and inflow (strong, head-only) baselines for the necessity result?
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

D = Path("/home/sangyu/Desktop/Master/SpecLens/outputs/class_fri/research_frontier")


def load(name):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else {}


def paired_p(a, b):
    """one-sided wilcoxon: is b < a (b better/lower hdel)?"""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    try:
        return float(wilcoxon(a, b, alternative="greater").pvalue), int((b < a).sum()), n
    except ValueError:
        return float("nan"), int((b < a).sum()), n


def main():
    attn = load("xmodel_attnloc_g50.json")
    fri = load("xmodel_friloc_g50.json")
    inflow = load("xmodel_inflowloc_g50.json")

    models = list(fri) or list(attn)
    print(f"{'model':12} {'loc':>5} | {'nec(attn)':>9} {'nec(fri)':>9} {'nec(infl)':>9} | "
          f"{'fri<attn p':>11} {'win':>6} | {'fri<infl p':>11} {'win':>6}")
    print("-" * 100)
    for mk in models:
        af = attn.get(mk, {})
        ff = fri.get(mk, {})
        nf = inflow.get(mk, {})
        na = af.get("nec_hdel", float("nan"))
        nfr = ff.get("nec_hdel", float("nan"))
        ni = nf.get("nec_hdel", float("nan"))
        # fri vs attn
        if af.get("nec_hdel_arr") and ff.get("nec_hdel_arr"):
            p_fa, w_fa, n_fa = paired_p(af["nec_hdel_arr"], ff["nec_hdel_arr"])
            fa = f"{p_fa:>11.2g} {str(w_fa)+'/'+str(n_fa):>6}"
        else:
            fa = f"{'-':>11} {'-':>6}"
        # fri vs inflow
        if nf.get("nec_hdel_arr") and ff.get("nec_hdel_arr"):
            p_fi, w_fi, n_fi = paired_p(nf["nec_hdel_arr"], ff["nec_hdel_arr"])
            fi = f"{p_fi:>11.2g} {str(w_fi)+'/'+str(n_fi):>6}"
        else:
            fi = f"{'-':>11} {'-':>6}"
        loc = ff.get("localization", "?")[:4]
        print(f"{mk:12} {loc:>5} | {na:>9.3f} {nfr:>9.3f} {ni:>9.3f} | {fa} | {fi}")

    print("\nlower nec_hdel = better. 'win' = #imgs where fri's nec_hdel < other's.")
    print("fri<infl: tests FRI-loc necessity vs REAL-inflow-loc necessity (head models).")
    # also report FRI map's OWN loc_hdel (sufficiency map = bad deletion map, expected high)
    print("\nFRI map's OWN loc_hdel (sufficiency map; high=expected) vs its nec_hdel:")
    for mk in models:
        ff = fri.get(mk, {})
        if ff:
            print(f"  {mk:12} loc_hdel {ff.get('loc_hdel', float('nan')):.3f} "
                  f"-> nec_hdel {ff.get('nec_hdel', float('nan')):.3f} "
                  f"(win {ff.get('win_del')}/{ff.get('n')})")


if __name__ == "__main__":
    main()
