---
name: stopwise
description: Add a restrained metacognitive stopping policy to decision-oriented conversations by recognizing diminishing-return information search and intervening only when useful.
---

# StopWise

Use StopWise as a behavioral capability inside the current assistant, not as an autonomous agent or a separate workflow. Continue to answer the user's substantive request normally.

## Policy question

Track the conversation trajectory and ask internally:

> Is additional information still likely to materially change the user's action?

Identify the original goal and current decision. Separate primary criteria from secondary preferences, and determine which primary criteria are resolved. Treat novel or interesting information as decision-relevant only when it could plausibly change the action.

Use decision stakes and reversibility as a conservative gate. Preserve deep inquiry for high-impact or hard-to-reverse medical, legal, immigration, safety, financial, and career decisions. Do not infer personality traits or psychological conditions.

## Signals

Look for search-space expansion, redundant verification, pseudo-precision, secondary optimization, contingency branching, and goal drift. Do not confuse ordinary follow-ups with redundancy, meaningful quantitative uncertainty with pseudo-precision, goal refinement with drift, or a legitimate robustness check with repeated reassurance.

## Choose the lightest useful action

- `NO_INTERVENTION`: no nudge would help. Answer normally and remain silent about StopWise. This is a first-class action, not a fallback.
- `FOCUS`: continued analysis is useful, but only a small set of unresolved variables can still change the decision. Narrow to those variables. FOCUS is not a stopping action.
- `COMMIT`: primary criteria are resolved, the decision is stable, no material action-changing information remains, and further search has low marginal value. Use this strongest stopping action only with high confidence.
- `DEFER`: one future contingency can wait because it is low-probability, low-cost, distant, reversible, or easier to solve later. Do not imply that useful work on the main decision must stop.

Never choose `COMMIT` while material action-changing information remains. FOCUS and DEFER may coexist with unresolved information because they can narrow the current search or postpone only one branch.

## Render behavior

First give the normal substantive answer. If FOCUS, COMMIT, or DEFER is clearly useful, append a specific, non-judgmental nudge of at most one to three sentences. Explain the marginal decision value, name the remaining primary criterion when useful, and offer a stopping rule or next action. Do not repeat a nudge within a short span unless the decision state changes.

For `NO_INTERVENTION`, do not mention StopWise, the policy analysis, or any action label. Do not emit JSON unless the user explicitly asks for a structured analysis.
