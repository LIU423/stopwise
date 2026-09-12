# Theory and design rationale

StopWise is a research-inspired engineering synthesis. It operationalizes observable conversational proxies for diminishing-return information seeking; it does not infer a person's literal internal stopping rule, reproduce either source framework, or diagnose “overthinking.”

## 1. Information-search stopping

Browne, Pitts, and Wetherbe study how people terminate information search in online tasks. Their results support two design premises used here: information search needs stopping criteria, and the rule used depends on task context. Related work by Browne and Pitts distinguishes information sufficiency in design problems from convergence in choice problems.

StopWise translates those ideas into a narrower engineering question: **could additional information realistically change the user's action?** The policy uses conversation-visible evidence—resolved criteria, remaining uncertainty, repeated checks, and search trajectory—rather than claims about internal cognition.

## 2. Adaptive metacognitive intervention

MetaCLASS frames metacognitive coaching as trajectory-conditioned action selection and explicitly represents non-intervention. Its educational taxonomy is not StopWise's taxonomy. The transferable design lesson is that a useful metacognitive system must decide both what intervention to provide and whether intervention is warranted at all.

That motivates a small policy with a genuine `NO_INTERVENTION` action. **Silence is a first-class action.** It also motivates evaluating false interventions and the precision and recall of silence rather than rewarding an assistant simply for producing a nudge.

## 3. StopWise synthesis

```text
Cognitive stopping research
        ↓
When is further information still useful?

Adaptive metacognitive intervention
        ↓
When should an AI intervene?

                ↓

             StopWise

When is further information unlikely to change the decision,
and should the assistant say anything about it?
```

| Research idea | StopWise design consequence |
|---|---|
| Information search requires stopping criteria | Estimate whether material action-changing information remains |
| Search can reach informational sufficiency | `COMMIT` represents low expected value from further search |
| Stopping depends on task context | Stakes and reversibility adjust intervention thresholds |
| Metacognitive support depends on trajectory | Analyze conversation history, not only the latest turn |
| Intervention is not mandatory | Treat `NO_INTERVENTION` as a first-class action |
| Models may over-intervene | Use conservative thresholds and evaluate false intervention explicitly |

The guiding principle is contextual, not rigid: **if additional information is unlikely to change the user's action, further search may have diminishing decision value.** Novel information is not necessarily decision-relevant information.

## 4. Policy actions

| Action | Meaning | Key boundary |
|---|---|---|
| `NO_INTERVENTION` | No metacognitive nudge is useful | Normal and high-stakes inquiry continues without a StopWise comment |
| `FOCUS` | Continue, but narrow to action-changing variables | May coexist with unresolved material information; it is not a stopping action |
| `COMMIT` | Primary criteria are resolved and the decision is stable | Prohibited while material action-changing information remains |
| `DEFER` | One future branch need not be solved now | Other useful analysis may continue |

## 5. Conversational signals

- `search_space_expansion`: alternatives multiply after viable options exist without a decision-relevant reason.
- `redundant_verification`: the same conclusion is rechecked without materially new evidence.
- `pseudo_precision`: requested precision exceeds the available measurement basis.
- `secondary_optimization`: low-impact differences dominate after primary criteria are satisfied.
- `contingency_branching`: a remote, low-cost, or reversible future branch consumes attention too early.
- `goal_drift`: the current subproblem no longer contributes materially to the original objective.

These are contextual observations, not diagnoses. The analyzer must preserve the distinctions between robustness and reassurance, measurable uncertainty and pseudo-precision, relevant and merely novel evidence, exploration and expansion, and goal refinement and drift.

## 6. Stakes and reversibility

StopWise uses **stakes × reversibility** as a conservative intervention gate.

- High-impact, hard-to-reverse medical, legal, immigration, safety, major financial, or career decisions warrant deeper analysis and a high threshold for `COMMIT` or `DEFER`.
- Medium-impact, reversible decisions suit bounded analysis.
- Low-impact, easily reversible decisions suit satisficing once primary criteria are met.

This gate adjusts intervention confidence; it does not estimate a user's personality or capacity.

## 7. Error asymmetry

The policy distinguishes four errors:

1. **Premature Commit:** `COMMIT` while material action-changing information remains.
2. **Unsafe Defer:** `DEFER` for a branch that matters to the decision now.
3. **False Intervention:** FOCUS, COMMIT, or DEFER when silence was preferable.
4. **Missed Intervention:** silence when a useful intervention was clearly warranted.

Their costs are context-dependent, but the core safety ordering is:

```text
premature COMMIT > unsafe DEFER > unnecessary FOCUS
```

A correct FOCUS is never classified as premature merely because useful information remains.

## 8. Limitations

StopWise is a prompt-driven prototype whose judgments inherit the underlying model's errors. It can misidentify goals, utilities, criteria, stakes, reversibility, or the value of missing evidence. It must not discourage necessary high-stakes information seeking. Its action labels are policy outputs, not evidence that a conversation or user has a psychological condition.

Policy-label agreement does not establish that StopWise reduces redundant turns or preserves decision quality. Those claims require controlled end-to-end comparison; see [Evaluation](evaluation.md).

## 9. References

- Browne, G. J., Pitts, M. G., & Wetherbe, J. C. (2007). Cognitive stopping rules for terminating information search in online tasks. *MIS Quarterly, 31*(1), 89–104. <https://doi.org/10.2307/25148782>
- Liu, N., Baraniuk, R., & Sonkar, S. (2026). MetaCLASS: Metacognitive coaching for learning with adaptive self-regulation support. *arXiv preprint arXiv:2602.02457* (v1). <https://doi.org/10.48550/arXiv.2602.02457>
- Browne, G. J., & Pitts, M. G. (2004). Stopping rule use during information search in design problems. *Organizational Behavior and Human Decision Processes, 95*(2), 208–224. <https://doi.org/10.1016/j.obhdp.2004.05.001>

Machine-readable entries are in [`references.bib`](../references.bib).
