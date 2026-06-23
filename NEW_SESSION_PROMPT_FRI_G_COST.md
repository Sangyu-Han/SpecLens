# New Session Prompt: FRI-G Cost Reduction (half-step + clean maps)

/goal

SpecLens repo에서 FRI-G (fri_alt + ridge-gate) 연구를 이어간다. 목표는 **순차 비용을 절반(16 steps)으로 줄이면서**, inflow 대비 hins−hdel gap의 **통계적 우위를 유지**하고, **점묘 없는 정성 품질**(composite/contrastive)과 **FRI spirit (model-agnostic)** 을 지키는 구성을 찾는 것이다.

## Current Continuation State (2026-06-15)

Start from these saved notes, not from the older baseline text below:

```text
outputs/class_fri/research_frontier/FRI_G_COST_REDUCTION_NOTES.md
outputs/class_fri/research_frontier/FRI_G_PAPER_INSPIRED_HANDOFF_20260615.md
outputs/class_fri/research_frontier/FRI_G_CHECKPOINT_20260616_BEFORE_NEC_SUFF_HH_READ.md
outputs/class_fri/research_frontier/FRI_G_CHECKPOINT_20260616_AFTER_CLPNS_NONLOW.md
outputs/class_fri/research_frontier/FRI_G_NEC_SUFF_HIDDEN_HERO_IDEAS_20260616.md
outputs/class_fri/research_frontier/FRI_G_CURRENT_SNAPSHOT_BEFORE_FANS_SNE_HH_20260616.md
outputs/class_fri/research_frontier/FRI_G_FANS_SNE_HH_EVENTMIX_IDEAS_20260616.md
outputs/class_fri/research_frontier/FRI_G_EVENT_RANKGAP_GATE_ANALYSIS_20260616.md
outputs/class_fri/research_frontier/FRI_G_MINI_EVENT_GATE_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_REDUCED_EVENT_PROBE_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_REDUCED_EVENT_PROBE_FIRST50_20260616.md
outputs/class_fri/research_frontier/FRI_G_REDUCED_SELECTOR_ANALYSIS_20260616.md
outputs/class_fri/research_frontier/FRI_G_CHECKPOINT_BEFORE_FANS_SNE_HH_REREAD_20260616.md
outputs/class_fri/research_frontier/FRI_G_FANS_SNE_HH_REREAD_IDEAS_20260616.md
outputs/class_fri/research_frontier/FRI_G_PNS24_DUAL_EVENT_VALIDATION_20260616.md
outputs/class_fri/research_frontier/FRI_G_PNS24_GUARD_CROSSSEED_20260616.md
outputs/class_fri/research_frontier/FRI_G_PHASE_AGREEMENT_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_CHECKPOINT_BEFORE_FANS_SNE_HH_REVISIT_20260616.md
outputs/class_fri/research_frontier/FRI_G_FANS_SNE_HH_REVISIT_IDEAS_20260616.md
outputs/class_fri/research_frontier/FRI_G_NECFLOOR_REDUCED_VALIDATION_20260616.md
outputs/class_fri/research_frontier/FRI_G_CHECKPOINT_AFTER_Q25_FRESH_AND_FANS_SNE_HH_REREAD_20260616.md
outputs/class_fri/research_frontier/FRI_G_PHASE_VOTE_REDUCER_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_LOGRHO_CONSISTENCY_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_RANKWINDOW_READOUT_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_GROUPCLOSURE_READOUT_SMOKE_20260616.md
outputs/class_fri/research_frontier/FRI_G_TV_SMOOTHNESS_DIAGNOSTIC_20260616.md
outputs/class_fri/research_frontier/FRI_G_TWOCHAIN_CONSENSUS_DIAGNOSTIC_20260616.md
outputs/class_fri/research_frontier/FRI_G_TWOCHAIN_CONS15_VALIDATION_20260616.md
outputs/class_fri/research_frontier/FRI_G_FANS_SNE_HH_SECOND_PASS_IDEAS_20260616.md
outputs/class_fri/research_frontier/FRI_G_PAUSE_BEFORE_EXTERNAL_RESULTS_20260616.md
```

Pause update (2026-06-16): after the user asked to stop and provide external
research results, the in-progress fresh `general50 --general-seed 126`
validation for `twochain_cons15 + nf25q25` was interrupted at 19/50 cases and
renamed to
`fri_g_tail_hero_top96_necfloor_q25_cons15_general50_seed126_partial19_interrupted_20260616.json`.
Do not treat it as full validation.  A WIP hook for event-valid redundancy
closure has been added to `research_fri_g_tail_hero_probe.py` as phase reducers
like `q25rc15` and `q25rc25`; it compiles and parses, but has not been
validated.  The pause state is saved in
`FRI_G_PAUSE_BEFORE_EXTERNAL_RESULTS_20260616.md`.

Latest checkpoint update (2026-06-16): the current state has been saved in
`FRI_G_TWOCHAIN_CONS15_VALIDATION_20260616.md`.  The preferred reduced metric
candidate is now `twochain_cons15 + nf25q25`
(`nf25q25:24:0|16:0:0.25:necfloor_suf:0+17+33:q25`).  It keeps the existing
`twochain_piece` pick but softly demotes the picked score by weak support from
the non-picked chain.  Four-surface aggregate over seed124 first50, seed125
first50, general50 seed123, and n100 first50: fixed q25 is
0.326965/0.918411/0.591445 hdel/hins/gap, while cons15+q25 is
0.323788/0.919501/0.595713.  Delta: dhdel -0.003177
(p_hdel_less=0.000480), dhins +0.001090, dgap +0.004268
(p_gap_greater=0.000609).  Qualitative seed42 panel improves red causality
from 19/20 to 20/20 and composite coherence from 0.5322 to 0.5344, but raw
FRI coherence is unchanged at 0.4843, so this is not a full pointillism
solution.  Before final promotion, run fresh general50 seed126 validation.
The second-pass paper read is saved in
`FRI_G_FANS_SNE_HH_SECOND_PASS_IDEAS_20260616.md`.  The next credible branch is
event-valid redundancy closure: combine FANS-style PN/PS event validity,
SNE-style super-necessity/stability under supersets, and Hidden-Heroes-style
coalition excess vs singleton bloat.  Do not import FANS SIR/data-neighborhood,
gradient subset optimization, SNE's generative reference, per-image TV mask
optimization, or Hidden-Heroes layer/gradient priors.

Latest reread update (2026-06-16): current research state has been saved in
`FRI_G_CHECKPOINT_BEFORE_FANS_SNE_HH_REREAD_20260616.md`.  After rereading
FANS, Sufficient/Necessary Explanations, and Hidden Heroes, the next preferred
research direction is a reduced dual-event PNS-lite probe, not another mini hard
gate.  Translate FANS to FRI-G as paired event validity only; do not import
FANS' SIR/data-neighborhood prior or gradient subset optimizer.  Translate SNE
as an explicit necessity-sufficiency axis: current `tail_event_probe` is mostly
necessity, so add a matching insertion/sufficiency event.  Translate Hidden
Heroes as `base FRI rank` vs `event-PNS causal rank`, not as gradient use.
Concrete first candidate: `top96_pns24c016r0_mix20` behind `free_middle`.
Expected cost is roughly 192 masks per fired seed-call, about 125 average masks
with the current free gate, versus 512 masks for full top96 event probing.
Success criterion: beat fixed `g24c016r0` on seed124 first20/first50 gap without
hdel regression, ideally closing at least half of the remaining gap to
free_middle full top96.
Validation update: implemented `--event-score pnslite` and optional
`name:groups:contexts:rhos:alpha:event_score` variant specs in
`scripts/research_fri_g_tail_hero_probe.py`.  The PNS-lite branch is positive
as a reduced **gap/sufficiency booster**, not as a reduced hdel branch.  On
seed124 first50, g24 is 0.2973/0.6261 hdel/gap, pns24a20 is
0.2983/0.6313; pns24a20 improves gap vs g24 by +0.00524 (p=0.0017) with hdel
regression +0.0010.  On seed125 first50, g24 is 0.3223/0.5852, pns24a25 is
0.3245/0.5907; pns24a25 improves gap vs g24 by +0.00554 (p=0.00075) with hdel
regression +0.0021.  Combined seed124+seed125 first50: g24 0.3098/0.6056,
pns24a20 0.3117/0.6105, pns24a25 0.3118/0.6107.  Do not replace g24 as the
hdel reduced arm.  Treat pns24a20/a25 as reduced gap/qual candidates and
selector sources.  Offline selector sweeps over PNS diagnostics do not yet
cross-validate beyond fixed pns, so no selector should be promoted.  Next best
step: qualitative panel comparing base, g24, pns24a20, pns24a25, and full
top96_mix20; then design a reliability guard that chooses g24 vs pns vs base
only if it validates on a fresh seed surface.  Qualitative update: guarded
g24/pns20/pns25 panels were generated on the seed42 20-case panel.  Full
top96_mix20 remains better on red causality (18/20 blue, 19/20 red, coh 0.535,
gap 0.591).  Reduced guarded g24 is 18/20 blue, 18/20 red, coh 0.536, gap
0.572.  Reduced guarded pns20 is 18/20 blue, 18/20 red, coh 0.537, gap 0.574;
pns25 is 18/20 blue, 18/20 red, coh 0.535, gap 0.574.  Reduced PNS does not
show an obvious new pointillism collapse, but it also does not recover full
top96's red-causality edge.  For reduced qualitative/gap work, prefer pns25
slightly over pns20; for reduced hdel, keep g24.
Guard update: `scripts/analyze_fri_g_pns24_crossseed_guard.py` now reproduces
cross-seed reliability-guard sweeps.  Results are negative for promotion:
same-surface and train/test threshold rules expose oracle headroom but do not
beat fixed pns20/pns25 robustly across seed124<->seed125.  Do not spend more
time on simple threshold selectors over the current diagnostics.  If pursuing a
guard, add a new reliability signal first, such as rank-movement agreement,
branch-harm detection, or an independent reduced group design for stability
checking.
Implementation note: `tail_event_probe` now writes `top_patches` and
`patch_effect_*` summaries into each event probe diag.  Existing artifacts do
not have these fields; future runs can use them to measure g24-vs-pns rank
movement agreement without storing full patch-effect arrays.
Independent group-phase update: variant specs now support
`name:groups:contexts:rhos:alpha[:event_score[:group_phase]]`.  `group_phase`
rotates the rank pool before the deterministic rank-lattice design, changing
group co-membership without adding any external prior.  Phase smoke:
seed124 first20 g24 phase0 0.3177/0.5976, g24 phase17 0.3203/0.5939,
pns24a25 phase0 0.3203/0.6009, pns24a25 phase17 0.3210/0.6001.  Seed125
first10 pns24a25 phase17 is promising (0.3322/0.5833 vs pns phase0
0.3349/0.5776).  However phase-overlap thresholds are not stable across
seed124->seed125, so do not deploy an overlap guard.  The useful insight is
that group co-membership variance is large; next credible experiment is
validating pns24a25 phase17 on seed124/seed125 first50 or testing a small
phase ensemble/minimax readout.
Revisit update: current research was checkpointed in
`FRI_G_CHECKPOINT_BEFORE_FANS_SNE_HH_REVISIT_20260616.md`, and the reread note
is saved as `FRI_G_FANS_SNE_HH_REVISIT_IDEAS_20260616.md`.  The main
conclusion changed from "combine necessity and sufficiency harder" to
"enforce necessity first, then use sufficiency as a constrained rerank/rescue."
Do not promote the current geometric-mean PNS-lite as the final reduced branch:
it improves gap through insertion but gives small hdel regressions versus g24.
The next implementation candidates are `fansweighted`
(`nec_score * live_event_prevalence + suf_score * weak_event_prevalence`) and
`necfloor_suf`/`pnslex` (sufficiency only reorders groups/patches that pass a
necessity floor).  Hidden-Heroes inspiration should be translated as
rank/effect redundancy and group-level superadditivity, not gradients.  SNE's
super-necessity suggests context/rho stability as the modality-general
anti-pointillism/OOD guard.
Necessity-floor validation update: implemented `fansweighted`, `necfloor_suf`,
and `pnslex` event scores in `scripts/research_fri_g_tail_hero_probe.py`; the
qualpanel CLI now accepts the same event scores.  `fansweighted` is useful as a
diagnostic but not as a fixed arm because seed125 first20 regresses hdel.
`necfloor_suf` is now the active reduced candidate.  Combined seed124+seed125
first50: g24 0.3098/0.6056 hdel/gap, necfloor24a20 0.3098/0.6076, and
necfloor24a25 0.3094/0.6074.  Paired vs g24: necfloor20 dgap +0.00199
(p=0.00035) with dhdel +0.00002; necfloor25 dgap +0.00181 (p=0.027) with
dhdel -0.00041.  Qual panel, same settings as prior g24/PNS panels:
necfloor20 blue/red 18/20 and 19/20, coh 0.5362, hdel/gap 0.3298/0.5733;
necfloor25 blue/red 18/20 and 19/20, coh 0.5332, hdel/gap 0.3274/0.5724.
This is the first reduced branch that improves gap over g24 without the PNS
hdel regression and also recovers full top96's red 19/20 qualitative count.
Next step: offline guard/selector over saved diagnostics, choosing between
g24, necfloor20, necfloor25, and base.  Do not run more alpha sweeps before
checking harmful-case diagnostics.
Selector smoke update: `fri_g_necfloor_selector_seed124_first50.json` and
`fri_g_necfloor_selector_seed125_first50.json` show same-surface headroom but
do not cross-check beyond fixed necfloor strongly enough.  Do not promote a
threshold guard yet.  Keep fixed `necfloor24a20`/`necfloor24a25` as candidates
and treat the selector results as harmful-case diagnostics only.
Broader validation update: `fri_g_tail_hero_top96_necfloor_general50_seed123_3seed.json`,
`fri_g_tail_hero_top96_necfloor_n100_first50_3seed.json`, and
`fri_g_tail_hero_top96_necfloor_multisurface_summary.json` are now saved.
Across seed124 first50, seed125 first50, general50, and n100 first50:
g24 0.3322/0.5841 hdel/gap, necfloor20 0.3323/0.5858, necfloor25
0.3314/0.5862.  Paired vs g24: necfloor20 dgap +0.00171 (p=4.1e-6) with
dhdel +0.00014; necfloor25 dgap +0.00215 (p=6.9e-5) with dhdel -0.00080
(p_hdel_less=0.00164).  On general50, necfloor25 is 0.3771/0.5420 vs g24
0.3784/0.5396; on n100 first50, necfloor25 is 0.3296/0.5880 vs g24
0.3306/0.5854.  Preferred fixed reduced candidate is now `necfloor24a25`.
It is still not a replacement for full `top96_mix20`, which remains materially
better on hdel/gap; this is a reduced-cost improvement over g24.
Phase-min checkpoint: current post-necfloor/phase-min state is saved in
`FRI_G_CHECKPOINT_AFTER_NECFLOOR_PHASEMIN_BEFORE_PAPER_REVISIT_20260616.md`.
Variant specs now also support
`name:groups:contexts:rhos:alpha[:event_score[:group_phase[:phase_reduce]]]`
with phase lists such as `0+17:min`.  This is still model-agnostic: it only
changes deterministic rank-lattice co-membership.  Combined seed124+seed125
first50: nf25p0 0.3094/0.6074 hdel/gap, nf25min 0.3073/0.6097.  Paired
nf25min vs nf25p0: dhdel -0.00210 (p_hdel_less=0.0178), dgap +0.00224
(p_gap=0.0795).  Broader phase-min validation is now saved in
`fri_g_tail_hero_top96_necfloor_phase_min_general50_seed123_3seed.json`,
`fri_g_tail_hero_top96_necfloor_phase_min_n100_first50_3seed.json`, and
`fri_g_tail_hero_top96_necfloor_phase_min_multisurface_summary.json`.
General50: nf25min 0.3756/0.5444 vs nf25p0 0.3771/0.5420.  N100 first50:
nf25min 0.3258/0.5904 vs nf25p0 0.3296/0.5880.  Across seed124 first50,
seed125 first50, general50, and n100 first50: nf25min 0.3290/0.5885 vs
nf25p0 0.3314/0.5862; paired dhdel -0.00238 (p_hdel_less=7.5e-7), dgap
+0.00231 (p_gap=0.00104).  `nf25min` is now the preferred reduced validation
candidate for qualitative testing, not a final method yet because it doubles
reduced event cost and still has tail regressions.
Paper revisit after phase-min is saved in
`FRI_G_PAPER_REVISIT_AFTER_PHASEMIN_20260616.md`.  The updated interpretation:
FANS supports event-valid PNS but not SIR/data-neighborhood or gradient subset
machinery; SNE supports "super" necessity/sufficiency as context/phase stability;
Hidden Heroes supports redundancy-rank diagnostics, not layer priors.  Immediate
next experiments: qualitative panel for nf25min, then try a three-phase
lower-quantile reducer (`0+17+33:q25`) before inventing another event score.
Qualitative phase-min update:
`qual_composite_piece_event_top96_necfloor24a25_phase_min_seed42.{png,json}` is
saved.  Same seed42 20-case panel: nf25min applied 15/20, blue 18/20, red
19/20, dBlue +0.244, dRed -1.246, coh 0.5303, hdel/gap 0.3244/0.5736.
This improves nf25p0 panel hdel/gap (0.3274/0.5724) but slightly worsens
coherence (0.5332 -> 0.5303) and does not visibly solve pointillism.  Worst
regressions vs nf25p0: nematode, lemur, altar, ctrl:goose.  Next experiment is
still `0+17+33:q25` to reduce strict-min brittleness.
Q25 update: tail probe now accepts phase reducers like `q25`.  Artifacts
`fri_g_tail_hero_top96_necfloor_q25_seed124_first50_3seed.json`,
`fri_g_tail_hero_top96_necfloor_q25_seed125_first50_3seed.json`, and
`fri_g_tail_hero_top96_necfloor_q25_seed124_125_first50_summary.json` are saved.
Combined seed124+seed125 first50: nf25p0 0.3094/0.6074 hdel/gap, nf25min2
0.3073/0.6097, nf25q25 0.3052/0.6129, nf25min3 0.3059/0.6120.  Paired q25 vs
min2: dhdel -0.00216 (p=9.96e-5), dgap +0.00324 (p=0.00117).  Paired q25 vs
p0: dhdel -0.00426, dgap +0.00548.  `nf25q25` is now the next preferred
validation candidate, but not final until general50/n100 and qualitative are
checked.
Q25 broader update: general50, n100 first50, multisurface, and qualitative
artifacts are saved:
`fri_g_tail_hero_top96_necfloor_q25_general50_seed123_3seed.json`,
`fri_g_tail_hero_top96_necfloor_q25_n100_first50_3seed.json`,
`fri_g_tail_hero_top96_necfloor_q25_multisurface_summary.json`, and
`qual_composite_piece_event_top96_necfloor24a25_phase_q25_seed42.{png,json}`.
Across seed124 first50, seed125 first50, general50, and n100 first50:
nf25p0 0.3314/0.5862 hdel/gap, nf25min2 0.3290/0.5885, nf25q25
0.3270/0.5914, nf25min3 0.3279/0.5901.  Paired q25 vs min2: dhdel -0.00202,
dhins +0.00089, dgap +0.00291, p_gap=6.37e-5, p_hdel_less=9.69e-7.  Q25
qualitative panel: applied 15/20, blue 18/20, red 19/20, dBlue +0.244, dRed
-1.340, coh 0.5322, hdel/gap 0.3255/0.5772.  This beats nf25min2 on gap and
coherence while preserving causality counts.  `nf25q25` is now the preferred
reduced candidate, but still not a full pointillism solution; next work should
be a diagnostic tail guard over q25 regressions, not more blind phase/alpha
sweeps.
Q25 tail-guard update: `scripts/analyze_fri_g_q25_tail_guard.py` and artifacts
`fri_g_q25_crossseed_guard_seed124_to_seed125.json`,
`fri_g_q25_crossseed_guard_seed125_to_seed124.json`,
`fri_g_q25_tail_guard_loso.json`, and
`fri_g_q25_tail_guard_nec_threshold_sweep.json` are saved.  I also extended
`analyze_fri_g_reduced_selector.py` with phase-effect features and generic
pairwise variant contrasts.  Strict leave-one-surface-out train-best threshold
guards do **not** beat fixed q25: seed125/general50/n100 heldouts are flat to
worse, so no learned threshold guard is promoted.  The most interpretable
post-hoc signal is `nf25min2_best_nec_score_max`: choose q25 if >=0.75 else
min2 gives aggregate 0.3265/0.5919 vs fixed q25 0.3270/0.5914 and nonnegative
gap on all four surfaces, but the gain is tiny and post-hoc.  Keep fixed
`nf25q25` as the preferred reduced candidate; next guard needs a stronger
pre-registered stability signal, not simple threshold mining.
Phase-score guard candidate update: `analyze_fri_g_reduced_selector.py` now
extracts phase-level stability features (per-phase score/effect stats and
top-patch overlap).  New artifacts:
`fri_g_q25_tail_guard_phase_loso.json`,
`fri_g_q25_tail_guard_phase_threshold_sweep.json`,
`fri_g_q25_phase_score_guard_loso.json`, and
`qual_composite_piece_event_top96_necfloor24a25_phase_score_guard_seed42.json`.
Fixed feature family `nf25min2_phase_score_mean_mean`: choose q25 if feature >=
threshold else min2.  When the threshold is selected on the other three
surfaces, heldout deltas vs fixed q25 are positive on all four surfaces
(seed124 dgap +0.00076, seed125 +0.00036, general50 +0.00008, n100 +0.00045).
Median LOSO threshold 0.2356 gives aggregate 0.3266/0.5919 vs fixed q25
0.3270/0.5914.  Offline qualitative row selection keeps blue/red 18/20 and
19/20, coh 0.5319, hdel/gap 0.3253/0.5774.  This is the first guard-shaped
signal that transfers when the feature family is fixed, but the gain is tiny and
it is still offline-only; do not replace fixed q25 until fresh-surface or
method-level validation.
Fresh q25/reread checkpoint: current state is saved in
`FRI_G_CHECKPOINT_AFTER_Q25_FRESH_AND_FANS_SNE_HH_REREAD_20260616.md`.
Fresh general50 seed126 validates fixed `nf25q25`: p0 0.3146/0.5950,
min2 0.3136/0.5962, q25 0.3117/0.6001, min3 0.3141/0.5962 hdel/gap.
Paired q25 vs min2 over 150 row-seed points: dhdel -0.00198
(p_hdel_less=0.0292), dgap +0.00391 (p_gap_greater=0.00603).  The previously
promising phase-score guard fails this fresh check: guard 0.3119/0.5996 vs
fixed q25 0.3117/0.6001, so keep it diagnostic-only and do not promote it.
After rereading FANS, SNE, and Hidden Heroes, the next branch should be an
offline hidden-hero/bloat diagnostic over saved q25 artifacts: compare base
rank, event rank, phase effect stability/range, suff bonus, and top-patch phase
overlap on helped vs harmful q25 cases.  Only promote a fixed bloat penalty if
it transfers across seed124, seed125, general123, n100, and fresh general126.
No InFlow prior, SIR/data-neighborhood prior, gradient optimizer, layer prior,
explainer, segmentation prior, attention prior, or representation prior.
Hidden-hero/bloat diagnostic update: added
`scripts/analyze_fri_g_q25_hidden_hero_bloat.py` and saved
`fri_g_q25_hidden_hero_bloat_diag.json`.  Over five q25 surfaces (250 cases),
q25 vs min2 is still positive overall: dhdel -0.00202, dhins +0.00110,
dgap +0.00311, clean_help_rate 0.424, harm_rate 0.336.  There is a weak but
real diagnostic signal: clean-help cases have lower `nf25q25_phase_top8_rank_range_std`
than harm cases (mean 4.74 vs 5.66), and hdel-gain correlates weakly with
phase/rank stability features.  But direct bloat fallback does not transfer:
median rule `q25 if phase_top8_rank_range_std <= 4.736 else min2` worsens
aggregate to 0.3244/0.5917 vs fixed q25 0.3239/0.5932, and LOSO threshold
search fails on seed124/general123/fresh126.  Keep hidden-hero/bloat features
diagnostic-only; do not promote a bloat penalty yet.
Phase-vote reducer smoke: implemented diagnostic phase reducers `qNNvK` and
`qNNsvK` in `research_fri_g_tail_hero_probe.py`.  `q25v2`/`q25sv2` require
top-quartile phase-effect support in at least 2 of 3 phase designs.  This was
tested as a model-agnostic anti-pointillism readout over the same q25 probes.
Seed124 first20: q25 0.3130/0.6066, q25v2 0.3160/0.6001, q25sv2
0.3142/0.6033 hdel/gap.  Seed42 qualitative: q25 blue/red 18/20 and 19/20,
coh 0.5322, hdel/gap 0.3255/0.5772; q25sv2 blue/red 18/20 and 18/20,
coh 0.5345, hdel/gap 0.3322/0.5620.  Visual read: pointillism remains.
Decision: do not promote phase-vote reducers; top-rank vote support is too
blunt and removes real deletion evidence.  Next anti-pointillism branch should
prefer smoother reliability or multi-level/log-erasure consistency, not hard
support gating.
Log-rho consistency smoke: tested partial-erasure consistency with
`rho=0,0.0625,0.25`, saved in
`FRI_G_LOGRHO_CONSISTENCY_SMOKE_20260616.md`.  Full q25+logrho (`q25lr3`) gives
a tiny repeatable insertion/gap boost but costs hdel and triples event-probe
cost: seed124 first20 q25 0.3130/0.6066 vs q25lr3 0.3134/0.6073; seed125
first20 q25 0.3076/0.5746 vs q25lr3 0.3080/0.5752 hdel/gap.  Combined delta:
dhdel +0.00041, dhins +0.00101 (p_greater 0.0265), dgap +0.00060
(p_greater 0.205).  Qualitative q25lr3 preserves blue/red 18/20 and 19/20 and
slightly improves hdel/gap on the seed42 panel, but it is not visibly less
pointillistic.  Cost-neutral replacement (`lr3p0`: one phase, three rhos) is
much worse than q25 on seed124 first20: 0.3190/0.5974 vs 0.3130/0.6066.
Decision: do not promote log-rho as a fixed branch.  Phase stability beats
erasure-level stability at equal cost; log-rho is only a diagnostic/small
insertion signal for now.
Rank-window readout smoke: implemented diagnostic phase reducers like `q25rw5`
and `q25rw9`, which smooth q25 phase effect over the current FRI/base-rank
source-pool order before `_mix_score`.  This is model-agnostic and adds no
forward cost, but it fails.  Seed124 first20: q25 0.3130/0.6066, q25rw5
0.3233/0.5955, q25rw9 0.3312/0.5871 hdel/gap.  Seed42 qualitative:
q25 blue/red 18/20 and 19/20, dRed -1.340, coh 0.5322, hdel/gap
0.3255/0.5772; q25rw5 blue/red 18/20 and 19/20, dRed -1.154, coh 0.5366,
hdel/gap 0.3353/0.5690.  Visual read: slightly smoother by proxy but still
scattered, with causality diluted.  Decision: do not promote rank-window
smoothing.  Coherence proxy can improve while causal deletion quality worsens.
Group-closure readout smoke: implemented diagnostic reducers `qNNgcmean`,
`qNNgcmax`, and `qNNgctop`, which reuse the same q25 probes but read patch
evidence from high-scoring coalitions containing the patch rather than
on-vs-off patch contrast.  Seed124 first20: q25 0.3130/0.6066, q25gcmean
0.3135/0.6064, q25gcmax 0.3185/0.6010, q25gctop 0.3247/0.5934 hdel/gap.
Seed42 qualitative: q25gcmean keeps blue/red 18/20 and 19/20 but slightly
weakens dRed/gap (dRed -1.311, hdel/gap 0.3258/0.5753 vs q25 dRed -1.340,
0.3255/0.5772) and does not visibly reduce pointillism.  Decision: do not
promote group-closure readout; `gcmean` is safe-ish diagnostic only,
`gcmax/gctop` are too broad.
TV smoothness diagnostic: qualitative panel visualizes raw FRI in column 2 and
event-updated gate/composite in columns 3-4, so event q25 cannot clean the raw
FRI pointillism.  Increasing existing `fri_solve` TV confirms pointillism is
partly an optimization smoothness issue, but direct TV destroys metrics.
Seed42 q25 baseline: blue/red 18/20 and 19/20, dRed -1.340, coh_fri 0.484,
coh_comp 0.532, hdel/gap 0.3255/0.5772.  TV=0.02: blue/red 18/20 and 19/20,
dRed -1.166, coh_fri 0.577, coh_comp 0.574, hdel/gap 0.3638/0.5394.
TV=0.02 with deletion_weight 0.9: blue/red 18/20 and 18/20, dRed -1.328,
coh_fri 0.575, hdel/gap 0.3518/0.5544.  TV=0.03 over-smooths and worsens
further.  Decision: do not promote higher TV; it proves the failure mode but is
not a solution.  Next direction should be modality-general optimization
stability/trajectory consensus or delayed-collapse scheduling, not another
post-hoc q25 readout smoother.
Two-chain consensus diagnostic: added qualitative-only `--chain-select
twochain_mean` and `twochain_min` to test seed-stability without spatial TV.
These reuse the two FRI solves already needed by `piece_hdel_posdh`.  Seed42
panel: piece q25 blue/red 18/20 and 19/20, dRed -1.340, coh_fri 0.484,
coh_comp 0.532, hdel/gap 0.3255/0.5772; twochain_mean keeps 18/20 and 19/20,
dRed -1.207, coh_fri 0.531, coh_comp 0.549, hdel/gap 0.3339/0.5584;
twochain_min keeps counts but weakens dRed -0.973 and hdel/gap 0.3350/0.5625.
Decision: do not promote simple mean/min consensus.  It is less destructive
than TV and confirms seed/trajectory stability is a better modality-general
direction, but wholesale averaging/intersection loses too much gap.  Next
credible branch should use seed consensus selectively as reliability/delayed
collapse support, not replace q25 wholesale.

Latest implemented branches:

- `scripts/research_fri_g_ranklattice_response.py`
- `scripts/research_fri_g_pns_lite_response.py`
- `scripts/research_fri_g_redundancy_response.py`

Current candidate base:

```text
twochain_piece selected s16t_gS
S=16 two independent chains, shared P64 ridge beta gate
score shape floor=0.3, temp=1.0
response branch uses deterministic rank-lattice groups over the FRI rank order
```

Important constraints from the discussion:

- Do not use InFlow as a prior or method component.  It is reporting-only.
- Do not use an external explainer, segmentation prior, or model-specific
  attention/representation prior.
- The active qualitative failure mode is pointillism: metric gains are not
  enough if maps collapse into scattered speckles.

Latest result:

```text
seed125 first50, clip convention:
InFlow                         hdel 0.2648  hins 0.8778  gap 0.6131
twochain_piece base            hdel 0.3352  hins 0.9086  gap 0.5734
rank response margin_mix20     hdel 0.3217  hins 0.9050  gap 0.5833
PNS-lite nec/uni25             gap ≈ 0.5833-0.5834
redundancy prefix_hero_mix20   hdel 0.3201  hins 0.9065  gap 0.5864

general50:
rank response margin_mix20     hdel 0.3742  hins 0.9161  gap 0.5419
redundancy prefix_debloat      hdel 0.3730  hins 0.9165  gap 0.5436

n100:
rank response margin_mix20     hdel 0.3097  hins 0.9090  gap 0.5993
redundancy prefix_hero         hdel 0.3076  hins 0.9095  gap 0.6019

seed124 full200:
rank response margin_mix20     hdel 0.3101  hins 0.9160  gap 0.6059
redundancy prefix_hero         hdel 0.3084  hins 0.9166  gap 0.6082

qualitative seed42 panel:
prefix_hero   blue 18/20, red 17/20, coh_comp 0.534, hdel/hins 0.3268/0.9129
prefix_debloat blue 18/20, red 18/20, coh_comp 0.534, hdel/hins 0.3298/0.9135

guarded structural selector:
if hdel_est < 0.161077 -> base
elif super_margin_mean >= -0.106907 -> prefix_hero
else -> prefix_debloat
general50/n100/seed124/seed125f50 dgap vs prefix:
+0.00274 / +0.00278 / +0.00456 / +0.00423
qual panel: blue 18/20, red 17/20, coh_comp 0.538, hdel/hins 0.3276/0.9036

red-safe guard candidate:
if hdel_est < 0.161077 -> base
elif super_margin_mean >= -0.05 -> prefix_hero
else -> prefix_debloat
general50/n100/seed124/seed125f50 dgap vs prefix:
+0.00255 / +0.00196 / +0.00448 / +0.00408
qual panel: blue 18/20, red 18/20, coh_comp 0.537, hdel/hins 0.3289/0.9030

CL-PNS low-hdel booster (current best reporting candidate):
if selected_chain_hdel_est < 0.1452 -> cheap CL-PNS clpiv_margin_mix20
else -> red-safe structural guard
cheap CL-PNS config: groups=32, group_size=16, pool_k=96,
del_contexts=0,16, ins_contexts=0,16, mix_alpha=0.20.
general50/n100first50/seed125first50/seed124full200/seed125full200 vs red-safe:
  hdel 0.3733->0.3717, gap 0.5445->0.5477, dgap +0.00329, p 0.0060
  hdel 0.3302->0.3282, gap 0.5862->0.5910, dgap +0.00476, p 0.0075
  hdel 0.3205->0.3195, gap 0.5874->0.5919, dgap +0.00446, p 0.00145
  hdel 0.3078->0.3068, gap 0.6104->0.6142, dgap +0.00380, p <1e-5
  hdel 0.3359->0.3352, gap 0.5740->0.5782, dgap +0.00424, p <1e-5
seed125 full200 vs InFlow: gap 0.5782 vs 0.5683, mean advantage +0.00995,
but p_gap_vs_inflow 0.359 and hdel remains worse (0.3352 vs 0.3071).
qual panel: blue 18/20, red 18/20, coh_comp 0.535, hdel/hins 0.3277/0.9163,
choices clpiv 4 / hero 2 / debloat 14.
Artifact: outputs/class_fri/research_frontier/fri_g_clpns_lowhdel_guard_eval.json

negative: subparts 4 -> 8 on n100 first50 does not recover n100 gap.
red-safe gap 0.5862 -> 0.5863 only; fixed hero 0.5885 -> 0.5887 only.
Do not run full n100 subparts=8 unless subgroup composition also changes.
```

Read: FANS/SNE-inspired PNS-lite is diagnostic but does not solve seed125.
Hidden-Heroes-style redundancy/superadditivity gives a small significant
increment over the previous rank response on seed125 first50, general50, n100,
and seed124 full200, but still does not reach InFlow on the fresh seed125
split.  Qualitatively, neither structural arm collapses into new pointillism;
`prefix_hero` is the better metric arm, while `prefix_debloat` preserves red
causality better on the panel.  The first guarded selector is the best metric
candidate and keeps coherence, but still drops red causality to 17/20. The
red-safe guard restores red 18/20 with a small metric concession and is the
current reporting candidate. Continue from the handoff's next-step section:
the CL-PNS low-hdel booster is now the best current reporting candidate.
Plainly increasing subgroup count to 8 was negative.  Do not spend more compute
on hero-only, naive PNS-product readouts, or the current `sched_suf16_nec32`
schedule.  The paper reread conclusion on 2026-06-16 was correct in a narrower
way: paired coalition-local event flips help, but only as a low-deletion-risk
insertion booster.  Seed125 full200 is now checked: the candidate beats
red-safe and has higher mean gap than InFlow, but does not reach significant
InFlow superiority and still has much worse hdel.  If continuing the bottleneck
investigation, focus on non-low-hdel hdel reduction rather than more low-hdel
insertion boosting.  First checks were negative: structural-arm selector search
only recovers about -0.0002 to -0.0005 seed125 full200 hdel, and CL-PNS on the
20 worst non-low hdel cases gives prefix/clpn/clpiv hdel 0.5699/0.5722/0.5726
against InFlow hdel 0.3076.  Moving response earlier to prefix_k=16 is also
negative: prefix16 margin20/35 hdel 0.5790/0.5802, clpn/clpiv unchanged around
0.572.  Therefore the next credible bottleneck direction is perturbation/
deletion stability and hidden-hero source discovery for non-low cases.  The
2026-06-16 reread of FANS/SNE/Hidden-Heroes updates the next-step priority:
first run a retention-shape audit on seed125 non-low hdel-bad20, then test
event-conditioned CL-PNS that only credits flips in valid factual event bands,
and finally try hidden-hero tail mining outside the top96 FRI rank pool if the
audit shows rank-source miss rather than pure hard-deletion collapse.
The retention-shape audit is now done:
`outputs/class_fri/research_frontier/FRI_G_NONLOW_RETENTION_SHAPE_RESULT_20260616.md`.
It shows rank-source miss, not hard-only collapse: hard hdel on bad20 is
InFlow 0.2770 vs redsafe/lowclpiv 0.5315, and soft rho=0.03 remains InFlow
0.2641 vs redsafe/lowclpiv 0.4978.  Next priority is hidden-hero source
expansion outside top96, with event-conditioned PN-style flips as the
acceptance test.
First tail-source smoke is also done:
`outputs/class_fri/research_frontier/FRI_G_TAIL_HERO_PROBE_RESULT_20260616.md`.
Naive tail96 promotion is negative (seed42 bad20 hdel/gap 0.6150/0.2246 vs
base 0.5712/0.3657).  Event-conditioned top96/full196 promotion has hdel
signal but loses insertion: top96_promote32 hdel 0.5548, gap 0.3568;
full196_promote32 hdel 0.5585, gap 0.3181.  Best-arm oracle over these arms
has hdel headroom 0.5712->0.5341 and gap headroom +0.0192, so next experiment
should be gentler event-conditioned readout: weak mix or promote8/16, with
insertion/sparse-risk guard; do not pursue tail96-only promotion without a
stronger acceptance test.
That gentler readout has now produced a positive branch:
`outputs/class_fri/research_frontier/FRI_G_EVENT_MIX20_RESULT_20260616.md`.
Seed125 first50, 3-seed merged:
base hdel/gap 0.3352/0.5734; top96_mix20 0.3103/0.5954; full196_mix20
0.3098/0.6000.  Paired vs base: full196_mix20 dhdel -0.02544
(p 4.54e-11), dgap +0.02662 (p 1.85e-9).  Bad20 3-seed also improves:
base 0.5804/0.3577 -> full196_mix20 0.5546/0.3808.  This is the first
credible non-low hdel rescue.  Validation update:
`outputs/class_fri/research_frontier/FRI_G_EVENT_MIX20_VALIDATION_20260616.md`.
general50 seed123 3-seed: top96_mix20 0.3678/0.5497, full196_mix20
0.3684/0.5516.  n100 first50 3-seed: top96_mix20 0.3179/0.5993,
full196_mix20 0.3228/0.5944.  Qual panel: both blue 18/20, red 19/20,
coh 0.535-0.537, zebra mirror preserved.  This now beats low-clpiv on
general50, n100 first50, seed125 first50, and qualitative.  Seed125 full200
top96_mix20 is also checked: base 0.3495/0.5614 -> top96_mix20
0.3264/0.5841; paired vs base dhdel -0.02313 (p 3.69e-33), dgap +0.02274
(p 5.53e-27).  Versus InFlow, mean gap is higher by +0.0158 but not
significant (p=0.1830), and hdel remains worse by +0.0193.  Seed124 full200
top96_mix20 is now checked: base 0.3204/0.5998 -> top96_mix20
0.2986/0.6211; paired vs base dhdel -0.02185, dgap +0.02135, and
p_gap_vs_inflow 6.53e-5.  This beats the old low-clpiv seed124 full200 result
0.3068/0.6142.  Preferred current reporting candidate is top96_mix20 because
it is safer on n100 and panel hdel; full196_mix20 remains a selector/oracle
source.  Still required: n100 full100 if desired, and adaptive-cost guard.  A
simple `selected_hdel_est >= 0.14` gate preserves most gains but fires on
70-88% of cases, so cost control needs a better event-validity/rank-gap
pre-screen rather than only hdel-est gating.
The first offline event/rank-gap gate analysis is now saved at
`outputs/class_fri/research_frontier/FRI_G_EVENT_RANKGAP_GATE_ANALYSIS_20260616.md`.
It found a better free middle-risk gate:
`selected_est_hdel >= 0.1168 and max(est_A_hdel, est_B_hdel) <= 0.7545`.
Case-level search on seed124+seed125 full400 fires 280/400 instead of 400/400,
with hdel 0.3147 and gap 0.6008, preserving 91.5% of the full top96_mix20 gap
gain.  The actual implementation gates per seed/case; deployment accounting is
774/1200 seed-call fires, hdel 0.3152, gap 0.6001, preserving 88.7% of the
full gain.  A first20 actual runner smoke
`fri_g_tail_hero_top96_mix20_free_middle_seed124_first20_3seed.json` matches
the per-seed offline composition exactly: 44/60 seed-call fires, hdel/gap
0.3094/0.6102 vs base 0.3270/0.5902 and full top96 0.3068/0.6128.  A 50%
free gate preserves only about 72.7% of the gap gain, so this does not yet
solve the half-cost goal.  Event-derived gates add only a small increment over
the free gate and require the event probe unless replaced by a cheap 8-12
group mini-screen.
`scripts/research_fri_g_tail_hero_probe.py` now implements this gate with
`--event-gate free_middle --gate-hdel-min 0.1168 --gate-hdel-max 0.7545`;
default behavior is unchanged (`--event-gate none`).  When the gate fails, it
skips the event probe and fills event arms with the base score.
The first mini event screen is also implemented as `--event-gate
free_middle_mini`, but it is not yet a reporting candidate.  Seed124 first20:
free_middle hdel/gap 0.3094/0.6102 with 44/60 full-event seed-calls and 235s;
free_middle_mini default hdel/gap 0.3134/0.6062 with 33/60 full-event
seed-calls and 212s.  Threshold simulation from the saved mini diagnostics
shows that preserving free_middle performance requires firing almost as often
as free_middle, while stronger skipping loses visible gap.  Keep mini screen
as exploratory unless group design or block scheduling improves it.
Mini direct readout was also tested with `--mini-mix-alphas 0.05,0.10,0.20`
and no full event probe:
`fri_g_tail_hero_top96_minimix_sweep_free_middle_seed124_first20_3seed.json`.
It is negative: base hdel/gap 0.3270/0.5902, full top96 0.3068/0.6128,
free_middle 0.3094/0.6102, but minimix05 0.3255/0.5912, minimix10
0.3252/0.5886, minimix20 0.3275/0.5830.  The mini patch-effect direction is
too noisy to replace full event response directly.  Do not spend more compute
on this exact 8-group rank-lattice mini design except to test a different
group design or block schedule.
The block schedule variant `--mini-blocks` was then tested:
`fri_g_tail_hero_top96_miniblock_sweep_free_middle_seed124_first20_3seed.json`.
It protects the base prefix and uses mini response only for the next block, but
still does not recover the full event gain: miniblock16_32_20 hdel/gap
0.3242/0.5907, miniblock24_48_20 0.3261/0.5908, miniblock32_48_20
0.3277/0.5901, miniblock32_64_20 0.3289/0.5889.  Conclusion after mini hard
gate, mini direct mix, and mini block schedule: this exact `groups=8,
contexts=0,16,rhos=0` mini design is not enough.  Next cost work should reduce
the full event probe itself, save full patch/group diagnostics for offline
selector learning, or test a different mini group design with better
rank-lattice coverage.
If continuing cost work, reduce CL-PNS overhead by making the implementation
explicitly adaptive-cost: compute CL-PNS only after the sparse hdel gate fires.
Reduced full-probe variants are now the more promising half-cost direction:
`outputs/class_fri/research_frontier/FRI_G_REDUCED_EVENT_PROBE_SMOKE_20260616.md`.
`scripts/research_fri_g_tail_hero_probe.py` supports `--variant-specs
name:groups:contexts:rhos:alpha`.  Seed124 first20 reduced probe smoke:
base 0.3270/0.5902, full top96 0.3068/0.6128, free_middle full top96
0.3094/0.6102.  Reduced arms: g16c016r0 0.3244/0.5904, g16c01632r0
0.3218/0.5933, g24c016r0 0.3177/0.5976, g16c016r01 0.3237/0.5910.
`g24c016r0` is the first cheap positive probe: about 96 masks versus 512 for
full event probe, with hdel/gap 0.3177/0.5976.  It is not enough alone, but
oracle over base+reduced arms reaches hdel/gap 0.3153/0.6040, so selector
headroom exists.  Next priority: validate g24c016r0/g16c01632 on a larger
surface and test nearby reduced variants (`g24 c0/16/32 r0`, `g32 c0/16 r0`,
`g24 c0/16 r0/0.1`).
First50 validation is now done:
`outputs/class_fri/research_frontier/FRI_G_REDUCED_EVENT_PROBE_FIRST50_20260616.md`.
Seed124 first50: base hdel/gap 0.3086/0.6164, full top96 0.2868/0.6377,
free_middle full top96 0.2889/0.6364.  Reduced arms: g16c016r0
0.3032/0.6185, g16c01632r0 0.3005/0.6201, g24c016r0 0.2973/0.6261,
g16c016r01 0.3029/0.6186.  g24c016r0 is the best single reduced arm:
paired vs base dhdel -0.01129, dgap +0.00965, p_gap 9.54e-6.  It preserves
about 48% of the free_middle gap gain with about 19% of full event-probe mask
count.  Oracle over base+reduced arms reaches hdel/gap 0.2957/0.6302, so the
next priority is selector analysis among base/g24c016r0/g16c01632r0 using
saved reduced-probe diagnostics, then validation on seed125/general surfaces.
Selector analysis is now done:
`outputs/class_fri/research_frontier/FRI_G_REDUCED_SELECTOR_ANALYSIS_20260616.md`.
Same-surface best two-rule improves g24 fixed slightly (g24 0.2973/0.6261 vs
two-rule 0.2963/0.6278), but first20/next30 cross-check does not robustly beat
fixed g24.  Current reduced reporting candidate remains fixed `g24c016r0`;
selector learning should wait for another validation surface.  Next priority:
validate fixed g24c016r0 on seed125 first50 or general50.

## Working Directory

```text
/home/sangyu/Desktop/Master/SpecLens
```

## Context (직전 세션 결과 — 모두 검증됨)

FRI-G = fri_alt 32-step solve (변경 없음) + 64 seeded pos-field probe states
(배치 forward 1콜) + side별 margin kernel-ridge → signed β + gate:

```text
score = n01(fri_final) × (0.1 + 0.9·σ(β/0.5))      ← metric용
map   = n01(fri_final) + 0.5·min(β, 0)             ← RdBu composite 표시용
option: per-class β → contrastive maps (추가 forward 0)
```

신호의 정체: logit-수준 상호억제 축 (softmax 정규화 아님), 전 밀도 존재,
기하 불요 (field는 분산 ~8배 절감 장치). 인과 검증: β-top16 삭제 →
margin −1.2~−2.4, β-bot16 삭제 → **+0.3~+1.4** (20케이스 중 18 통과).

## Cost Accounting (정확히 이 정의를 쓸 것)

```text
S = sequential solve steps (각 1 fwd + 1 bwd; 현재 32)
P = parallel probe forwards (배치 1-2콜; 현재 64)
aux = irr(1f+1b) + full/base(2f) ≈ 4 states (모든 구성 공통, 별도 표기)
F_total = S + P  (총 forward 횟수; backward는 S + 1)
```

주 목표: S ≤ 16, P ≤ 64 (exploration P ≤ 128 허용, 최종 후보는 ≤ 64).
probe는 배치라 wall-clock ≈ solve의 ~5%지만, P도 명시 보고할 것.

**Ultimate tier (최종 목표, 달성 불확실)**: `F_total = S + P ≤ 32` —
probe 포함 총 forward가 기존 fri_alt-32의 solve forward 수와 같거나 적게.
후보 분할 예: S=16+P=16 / S=20+P=12 / S=24+P=8. 이 체급에서 아래
Minimum 기준(metrics+qualitative)을 충족하면 별도 헤드라인으로 보고.
이를 위해 **P ∈ {8, 16} probe 곡선 측정이 1순위 실험**이다 (β가 P=16에서
살아남는지 미지; rank-target/λ/ktemp 튜닝과 묶어서).

## Baselines (clip convention; 이 수치를 그대로 비교 기준으로)

```text
general50 (무작위 일반 분포, seed 123 목록):
  inflow   hdel 0.3721  hins 0.8690  gap 0.4969
  s32      hdel 0.4103  hins 0.9139  gap 0.5036  (p=0.57 n.s.)
  s32_g    hdel 0.3818  hins 0.9232  gap 0.5414  (paired p=0.040)  ← 현재 최고
  s16      hdel 0.4597  hins 0.8945  gap 0.4348
  s16_g    hdel 0.4254  hins 0.9150  gap 0.4896  (불충분)

n100 (failure set, softplus_failure_scan_n100.csv):
  inflow   hdel 0.3104  hins 0.8616  gap 0.5512
  s32_g    hdel 0.3344  hins 0.9113  gap 0.5769  (p=0.11)
  s16_g    hdel 0.3538  hins 0.9074  gap 0.5536
```

3-seed 분산: 12케이스 세트에서 hdel σ ≈ 0.02 — **모든 주장은 seeds
{42,43,44} 최소 3개로**. (이전 세션 교훈: 단일 시드 s16≈s32는 운이었음.)

## Numerical Targets (모두 clip convention, 3-seed mean)

Minimum success (S≤16, P≤64):

```text
general50: gap ≥ 0.4969 (inflow mean 이상), hins ≥ 0.90, hdel ≤ 0.43
n100:      hdel ≤ 0.3650
qualitative: blue-causality ≥ 16/20 (아래 프로토콜)
```

Good success:

```text
general50: gap ≥ 0.52, paired wilcoxon p < 0.10, hdel ≤ 0.40
n100:      hdel ≤ 0.355
coherence(composite) ≥ coherence(fri_alt), 3-seed hdel std ≤ 0.02
```

Strong success (= s32_g를 절반 비용으로 재현):

```text
general50: gap ≥ 0.5414, p < 0.05, hdel ≤ 0.385, hins ≥ 0.92
n100:      hdel ≤ 0.335
qualitative: blue-causality ≥ 18/20, contrastive zebra mirror 유지
             (β_zebra: ZEB영역 ≥ +0.8, ELE영역 ≤ −0.2)
```

Stretch (둘 중 하나):

```text
(a) S≤16, P≤32로 Good 달성
(b) S=32 유지하되 g50 gap ≥ 0.56 (더 나은 Pareto 점)
```

**Ultimate (최종 티어 — 가능성 불확실, 달성 시 최우선 보고):**

```text
F_total = S + P ≤ 32 (aux 4 별도)에서 Minimum 기준 전부 충족:
  general50 gap ≥ 0.4969, hins ≥ 0.90, hdel ≤ 0.43
  n100 hdel ≤ 0.3650, blue-causality ≥ 16/20
실패해도 F_total별 (32/48/64/96) 성능 곡선을 산출해 Pareto 프런티어를
보고할 것 — "어디까지 내려갈 수 있는가" 자체가 논문 기여.
```

최종 후보는 **general200** (새 목록, random seed 124)으로 headline
p-value 확증할 것 — n=50의 p=0.04는 약한 증거다.

## Qualitative Targets (점묘 금지 — 정량화된 기준)

1. **blue-causality 패널** (`research_composite_general.py`의 20케이스):
   β<−0.5 패치 삭제 → margin 상승 비율 ≥ 16/20 (strong: 18/20)
2. **coherence**: pos-kernel smoothed 자기상관 — composite ≥ fri_alt
   (현재 0.50 vs 0.45)
3. diag8 composite 그림을 현재 버전(qual_composite_general.png)과
   나란히 렌더해서 시각 회귀 없는지 확인
4. **contrastive 옵션 보존**: probe 설계를 바꾸더라도 상태별 full logits
   기록을 유지해 per-class β가 추가 forward 0으로 가능해야 함

## FRI Spirit Constraints

- training-free, per-instance, no external segmentation
- 새 컴포넌트에 2D grid prior 금지. pos-kernel(=baseline 토큰 코사인)은
  "모델 자신의 위치 코드"라 허용 (legacy fri_alt 내부의 grid-TV는 기존
  방법의 일부로 현상 유지 — 단, 이를 ktv 등으로 빼는 실험은 환영)
- model-agnostic: mask→logits oracle + autograd만 사용. 삭제 연산은
  block0 embedding 치환 (baseline = pos-embed-only) 유지
- transformer-일반성 보존: RoPE 모델 폴백 (중립 content 치환 + RoPE가
  위치 공급; kernel은 RoPE 위상 거리) 이 막히는 설계 금지
- tabular 경로 보존: β가 signed attribution으로 직접 출력 가능해야 함

## Existing Files To Read First

```text
outputs/class_fri/research_frontier/NSD_FRI_NECESSARY_SET_NOTES.md   (FRI-G 섹션)
scripts/run_fri_gate_validation.py    (검증 하니스: arms/clip/general50/n100)
scripts/research_fri_halfstep.py      (seeded probes + ridge + clip_aucs + step sweep)
scripts/research_fri_fieldsolve.py    (fri_alt 정확 복제 + 주입 훅들)
scripts/research_kernelp_signal.py    (신호 해부 기계)
scripts/research_signal_verify.py     (검증 배터리: split-half/shift-null/인과)
scripts/research_composite_general.py (20케이스 정성+인과 패널)
```

결과 JSON: `outputs/class_fri/research_frontier/fri_gate_val_{general50,n100}.json`,
`fri_halfstep_sweep*.json`.

## Known Negatives (새 근거 없이 재시도 금지)

- s12 solve: 죽음 (hdel +0.05~0.09)
- 16-step 비용은 실재 (+0.035, 3-seed) — 단일 시드 동률에 속지 말 것
- field-gated del steps: 양극적 (competitor 케이스 파괴)
- field-DIM init (32-state): 노이즈로 solve 오염
- solve 자기 상태 ridge: 3중 반증 (풀링/페어/인접차분 — 옵티마이저가
  설계행렬 다양성을 죽임). solve 상태만으로 β 만들기는 구조적 불가
- split-half β 신뢰도 게이트: 16-state 반쪽에선 발화 0% (prior 지배)
- hard-signed score (음수 대역 추방): 64 probes에선 gate에 패배
  (0.55-0.59 vs 0.465) — 부호는 demote까지만
- 2×14-step 평균(ens2, 세션2): 약함. 단 **선택**(selection)은 미시험
- lean64식 3-point prefix 선택: 추정 불안정

## Promising Directions (작고 결정적인 실험 우선)

1. **16-step 전용 하이퍼 재튜닝** — 한 번도 안 함. 32-step용 cosine
   (lr 0.45→0.01), l1 0.003, irr 0.05, dw 0.5를 그대로 썼다. 16-step의
   +0.035 갭 일부는 스케줄 아티팩트일 가능성. lr/lr_end/dw/l1/grid-TV
   weight 스윕 (12케이스 × 3 seeds로 빠르게).
2. **β-주도 irrelevance**: irr은 load-bearing (제거 시 +0.037, 세션2).
   irr := mix(1/gradnorm, n01(−β))로 교체하면 적은 스텝으로 수렴할 수도.
   (β-init은 실패했지만 irr 채널은 미시험 — 다른 주입점.)
3. **probe-count 곡선 (Ultimate tier의 관문)**: gate 품질 vs
   P ∈ {8,16,32,48,64}; rank-target ridge (+0.08 관찰), λ/ktemp 튜닝
   포함. P=32면 stretch (a), P=16이 살면 Ultimate 사정권. 저-P에서는
   ridge prior 강화(λ↑, ktemp 조정)와 dual-side 통합 적합(상태 수 2배
   효과) 같은 추정 효율 레버를 함께 시험할 것.
4. **2×16-step 체인 배치 실행 + 케이스별 선택**: batch-2 forward로
   wall-clock ≈ 1체인, 순차 16 유지. 선택은 probe est (충분히 촘촘한
   grid로 — lean64 교훈). 세션 전체에서 selection > averaging이었다.
5. **혼합 설계 ridge**: probes + solve 상태를 합쳐 적합 (collinearity
   희석 여부 싸게 확인).
6. **soft 신뢰도 가중**: gate 강도를 |β| 분위수나 split-half r로 연속
   조절 (binary는 죽었지만 soft는 미시험).
7. probe 상태를 MAS-calibration 앵커로 겸용 (비용 공유).

## Evaluation Protocol

```text
convention: clip 단일 (clip_aucs)
dev loop:   12케이스 (diag8 + ctrl idx 1,12,36,39) × seeds {42,43,44}
final:      general50 (seed 123 목록) + n100 CSV + general200 (seed 124)
검정:        paired wilcoxon one-sided (gap vs inflow)
정성:        20케이스 composite 패널 (blue-causality + coherence + 그림)
```

## Required Reporting

```text
config (S, P, 하이퍼) / 3-seed hdel·hins·gap mean±std (양 세트)
paired p-value / win_del·win_ins·win_both
blue-causality x/20, coherence, diag8 composite 그림
contrastive zebra mirror 수치
worst 5 regressions vs s32_g
negative results 기록
```

## Autonomy

Work autonomously. 제안에서 멈추지 말고 구현→실행→검사→반복.
방향이 죽으면 negative로 기록하고 피벗. research script 먼저, core 수정은
검증 후. 모든 주장에 3-seed.
