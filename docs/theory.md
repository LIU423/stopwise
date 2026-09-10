# Theory and design rationale

StopWise operationalizes a narrow question: **could unresolved information realistically change the user's action?** It does not estimate whether a person is “overthinking,” and it makes no psychological diagnosis.

## Cognitive stopping rules

Browne, Pitts, and Wetherbe (2007) describe cognitive stopping rules used to terminate information search in online tasks. StopWise borrows the practical intuition that search can stop when sufficient primary information has been acquired and further evidence no longer changes the working representation of the decision.

The middleware does not claim to recover a person's internal stopping rule. It detects conversational proxies: repeated verification, expanding a candidate set after viable options exist, unsupported numerical precision, optimization of secondary attributes, premature contingency planning, and drift from the original goal.

Reference: Browne, G. J., Pitts, M. G., & Wetherbe, J. C. (2007). Cognitive stopping rules for terminating information search in online tasks. *MIS Quarterly, 31*(1), 89–104.

## Adaptive metacognitive intervention

StopWise is also motivated by trajectory-aware metacognitive coaching, including the MetaCLASS framing. The relevant design lesson is that intervention is an action choice rather than a mandatory response. A system that always produces coaching language has a compulsive-intervention bias.

Accordingly, `NO_INTERVENTION` is a first-class and default action. Silence is useful behavior when inquiry remains decision-relevant or when a nudge has no clear benefit.

## Policy

The four actions form a deliberately small policy:

| Action | Decision state | Intended effect |
|---|---|---|
| `NO_INTERVENTION` | Important uncertainty remains, or evidence is ambiguous | Preserve useful inquiry |
| `FOCUS` | Analysis still helps, but only a few variables can alter the choice | Bound the search |
| `COMMIT` | Primary criteria are resolved and the choice is stable | Offer a stopping rule |
| `DEFER` | A future branch is cheap or easy to handle later | Delay low-value optimization |

Risk is approximated through decision stakes and reversibility. High-stakes, hard-to-reverse decisions receive the highest threshold for intervention. Low-stakes, easily reversible decisions are better suited to satisficing once primary criteria are met.

## Asymmetric error costs

The main safety objective is:

```text
Cost(premature intervention) > Cost(missed intervention)
```

A premature intervention occurs when StopWise recommends `FOCUS`, `COMMIT`, or `DEFER` while key action-changing information still needs investigation. A missed intervention occurs when it remains silent after search has clearly entered diminishing returns. The prompt and schema therefore prevent `COMMIT` whenever `unresolved_action_changing_information` is true and otherwise bias ambiguous cases toward silence.

## Evaluation direction

Initial evaluation uses handcrafted and semi-synthetic trajectories. Action accuracy is useful but insufficient; premature and missed intervention rates should be reported separately. A later controlled study can compare a baseline model with a model plus StopWise on decision accuracy, total turns, redundant turns, premature interventions, and the fraction of queries that reveal action-changing information.

