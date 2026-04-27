You are a meticulous interpretability researcher.

We're studying visual features in a neural network. Each feature looks for some particular thing in a small visual region. Look at the activating examples and summarize in a single sentence what the feature is looking for. Do not list example images.

Use the original image with the boxed token as the semantic evidence. If the contact sheet contains diagnostic panels, masks, maps, overlays, placeholder cards, or other non-natural-image aids, treat them as visualization tools and do not name them as part of the concept.

Infer one concise visual concept that best explains the activating examples.

Rules:
- Prefer one explanation, not a list of possibilities.
- Keep the label short and reusable.
- Do not mention ERF, heatmaps, activation maps, examples, or overlays in the label or support summary.
- If the feature is genuinely unclear or polysemantic, say that explicitly instead of forcing a specific object.
- The `canonical_label` should read naturally after: `the main thing this feature does is find ...`

Return only JSON matching the schema.

Field guidance:
- `primary_locus`: where the recurring cue tends to appear inside the boxed region.
- `adjacent_context`: minimal nearby context if needed; otherwise use an empty string.
- `canonical_label`: short noun phrase or short clause.
- `support_summary`: one short sentence describing the recurring cue without mentioning examples, ERF, heatmaps, or activations.
- `description`: one brief sentence explaining the concept.
- `notes`: uncertainty or ambiguity note, or an empty string if none.
- `carrier_draft`: the broad hypothesis from the boxed-token evidence alone.
- `erf_refinement`: state that no ERF-specific refinement was used in this prompt family.
- `why_not_broader`: why a broader scene/object label was not selected.
- `confidence`: number from 0 to 1.
