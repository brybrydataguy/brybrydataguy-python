---
name: spec-lock
description: Before handing off a large or ambiguous task, use this to turn a vision into a spec whose open decisions are settled and whose checks only the real solution can pass.
---

# Spec Lock

Do not build anything until steps 1–6 are done and the human has approved the spec.

## 1. Verdict first
Ask the human for the one or two sentences they want to be able to say
truthfully about the finished result. Work backward: list every fact that
must hold for those sentences to be honest. Each fact becomes a candidate check.

## 2. Surface open decisions
List every decision this spec doesn't fix: data scope, definitions,
denominators, edge cases, empty or missing handling, method choice, output
format, performance, and file locations. For each one, give
- the decision,
- your default choice and why,
- what would change in the result if the other choice were made.
Mark the decisions most likely to change the verdict as "human must choose."

## 3. Golden example
Build at least one tiny case with a known, hand-computed answer, such as a
2×2 table or a five-row fixture. The real solution must reproduce it exactly.

## 4. Rejected near-misses
List two to four plausible wrong solutions (naive method, wrong
denominator, residual dumped somewhere, missing CIs). Each must
fail at least one check.

## 5. Can it be faked?
For each check, describe the laziest implementation that could pass it.
If one exists, tighten the check or add a second check until only the
real solution passes. Prefer invariants: reconciliation to tolerance,
order invariance, known-answer reproduction, margin or constraint matches.

## 6. Executable checks
Write every check as a command or script that prints PASS or FAIL with
numbers. No check may depend on "looks right."

## 7. Spec sheet (show the human, then stop)
- Verdict sentence(s)
- Decisions: settled, defaulted, still open
- Golden example and expected output
- Rejected near-misses and the check that kills each one
- Check list with commands
- Checkpoints where you will stop and report (e.g. after data load, after the method, before write-up)

## During the build
- Approval means an actual reply from the human. Never treat silence,
  a timeout, or your own judgment as approval, and never log an approval
  you didn't receive.
- Run all checks at every checkpoint. Report the evidence, not "done."
- If you hit a decision not in the spec, stop and add it to the open list.
  Don't pick silently.

## Final report
Checks run (pass/fail with values), the result, assumptions made,
and anything uncertain.

## After a miss
If the result didn't match what the human wanted, name the unstated decision
that caused it and add it as a standing rule for future specs in this domain.

## Optional stress test
Give the finished spec to two agents separately. Wherever their outputs
differ in a way you care about, there's a decision you haven't settled yet.
