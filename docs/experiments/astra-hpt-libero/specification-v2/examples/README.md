# Proposed contracts and prompt

These files are planning fixtures, not validated runtime schemas or evidence of generated scenes.

- [learner_report.json](learner_report.json): a fabricated training-control report showing the fields an adaptive decision consumes. All measurements are illustrative.
- [adaptive_decision.json](adaptive_decision.json): valid allocation arithmetic across five templates. Scene/task references are placeholders requiring actual compilation and witness validation.
- [uniform_decision.json](uniform_decision.json): the matched uniform allocation, with five positive template quotas and no learner report.
- [generator_prompt.md](generator_prompt.md): the proposed provider instructions and separation of adaptive versus uniform inputs.

The examples demonstrate the JSON interface only. A completed implementation must validate it using versioned typed schemas, resolve all referenced artifacts, enforce geometry/novelty/attempt budgets, and reject placeholders. See the [machine manifest](../implementation_manifest.yaml) for numerical authority.
