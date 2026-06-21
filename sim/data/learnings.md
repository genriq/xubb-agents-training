# Learned principles

Auto-distilled from Self-Improve runs. Stylistic lessons auto-inject into the generator + optimizer prompts once support ≥ 2.

## Structural — suggested lint rules (promote to code by hand)
- **Ensure agents adhere strictly to their designated roles to prevent off-role behavior.** _(support 1, avg +7.0, ACTIVE)_
- **For real-time agents, trigger only on explicit evidence in the newest input and ignore stale or resolved prior context.** _(support 1, avg +5.0, ACTIVE)_
- **Role-specific agents should verify the latest speaker/source before acting, rather than inferring whose problem to solve from surrounding context.** _(support 1, avg +5.0, ACTIVE)_
- **Use cooldowns and unresolved-problem checks for agents prone to repeated advice, so they remain sparse without missing fresh issues.** _(support 1, avg +6.0, ACTIVE)_
- **Gate advice on the most recent explicit relevant cue, and suppress responses based only on older context or conversational transitions.** _(support 1, avg +16.0, ACTIVE)_
- **Define each agent’s lane with speaker/source constraints and non-overlap rules so agents do not respond to the wrong participant or duplicate another agent’s job.** _(support 1, avg +15.0, ACTIVE)_
- **Tune cooldowns by task urgency: same-turn issues need permissive firing, while repetitive logistics or status checks need stronger cooldowns to avoid stale repeats.** _(support 1, avg +16.0, ACTIVE)_
- **For commitment-tracking agents, require explicit owner-plus-action evidence and forbid inferred details or overlap with planning/scheduling advice.** _(support 1, avg +16.0, ACTIVE)_
- **Trigger agents only from explicit signals in the latest relevant external turn; do not let prior context or user-originated side remarks activate real-time advice.** _(support 1, avg +10.0, ACTIVE)_
- **Give each agent narrow role boundaries with explicit exclusions so it cannot drift into adjacent coaching responsibilities.** _(support 1, avg +10.0, ACTIVE)_
- **For real-time copilots, tune cooldowns around immediacy: allow same-turn responses to clear direct triggers while preventing delayed stale whispers.** _(support 1, avg +2.0, ACTIVE)_
- **Constrain rewrite or repair agents to a single concrete output form to prevent them from becoming generic advice generators.** _(support 1, avg +2.0, ACTIVE)_

## Stylistic — injected into prompts when ACTIVE
- **Reduce whisper frequency by increasing cooldown periods and emphasizing criticality to maintain high signal-to-noise ratio.** _(support 1, avg +6.0, candidate)_
- **Ensure whispers provide unique, high-value insights to reduce noise and improve signal quality.** _(support 1, avg +7.0, candidate)_
- **Capture tasks or constraints only when the user makes a new, explicit commitment or states a quoteable constraint; avoid inventing actions from vague plans.** _(support 1, avg +6.0, candidate)_

## Domain-specific
_(none yet)_
