You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Find the best reusable feature label first, then optionally produce short token-level handoff notes for later evaluation.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these SAE-fire images.
- Keep that canonical label fixed. Do not narrow or rewrite it just to satisfy token-local explanation fields.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- After choosing the canonical label, explain briefly why that label is the best shared interpretation in `rationale`.
- Then optionally add any of the following only if they are genuinely useful and supported:
  - `support_summary`: one short sentence about what is consistently visible across examples
  - `target_cue`: the most useful local cue for token-level matching
  - `adjacent_context`: nearby context that repeatedly helps disambiguate the cue
- If one of those optional fields is not clearly supported, leave it empty rather than forcing it.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- support_summary
- target_cue
- adjacent_context
- confidence
