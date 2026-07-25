# Design notes

Background reasoning for the debugging agent. `CLAUDE.md` has the rules that
bind day-to-day work; this file has the *why*, so decisions aren't relitigated.

## Deterministic playbooks vs. autonomous agent

A false binary. The real question is which layer is deterministic.

Playbooks serve one incident class each, so maintenance cost scales with the
number of failure modes — unbounded, and they rot silently. A free-form agent
avoids that but wanders, anchors on the first plausible story, and gathers only
confirming evidence.

Resolution: **agentic planning, deterministic tools.** A good tool serves every
investigation touching that data source, so maintenance scales with the number
of data sources — small and stable. Novel incidents still work because nobody had
to anticipate them.

## Four layers

| Layer | What it is | Change rate |
|---|---|---|
| Harness | Loop control, dispatch, validation, budgets, evidence ledger, replay/eval | Rarely |
| Tools | Data access and analysis (this repo) | With data sources |
| Skills | Domain semantics, incident-class priors, heuristics | Constantly |
| Agent | Orchestrator prompt, sub-agent prompts, output schema | Rarely, eval-gated |

Roughly 80% harness and tools, 15% skills, 5% agent. Teams routinely plan this
backwards.

**Skills are the contribution surface.** After a postmortem an SRE writes a note
— "Paxos leader election storms show up as `raft.election_count` spiking with flat
QPS; the tell versus a network partition is `heartbeat_rtt` p50 staying normal
while p99 doesn't" — and the agent retrieves it as a *prior* when the symptom
matches. Nobody maintains an execution graph. Skipping a contribution degrades
the system gracefully instead of breaking it.

Keep skills declarative. The moment someone writes "step 1, step 2" into a skill,
we've reinvented the playbook and inherited its maintenance cost.

## Anti-hallucination scaffolding

- **Closed-vocabulary tool calls.** No raw query strings, ever. Structured params
  validated against a live schema registry.
- **Empty vs. invalid must be distinguishable.** See invariant 1 in `CLAUDE.md`.
- **Evidence ledger.** Every claim cites a result ID. A verifier pass — fresh
  model call, sees claims and raw results but *not* the agent's reasoning trace —
  strips unsupported claims rather than softening them.
- **Mandatory falsification.** Before concluding, the agent states what it would
  expect if its leading hypothesis were wrong, and checks that. "If it's a bad
  rollout, onset aligns with rollout timestamps in rolled cells and not in
  un-rolled ones."
- **Mandatory opening sweep.** A fixed parallel fan-out before any drill-down.
  Costs nothing in flexibility, prevents tunnel vision.
- **Explicit "not checked" and "ruled out" lists.** Often more useful to on-call
  than the conclusion.
- **Self-consistency on P0s.** 3–5 independent branches; divergence means surface
  multiple hypotheses rather than picking one.
- **"I don't know" is a first-class, rewarded output.** If the eval rewards
  producing an answer, the system learns to fabricate one.

## Metrics-specific decisions

- **Semantic catalog is a real investment.** Auto-draft descriptions from the
  registry, have owners review. The agent is close to useless without good
  descriptions, and it's the piece teams underinvest in.
- **Field values resolve through a tool, never from model memory.**
- **Progressive narrowing** over full-cardinality queries: global → region →
  cell → job → task.
- **Contribution analysis beats agent-driven scanning.** One `explain_delta` call
  replaces fifteen group-by queries and does the arithmetic in numpy.
- **Change point in code, not in the model.** The onset timestamp is the highest-
  value single fact in an incident — it's what you intersect with the deploy log.
- **Peer baselines beat temporal baselines.** "8x the median of its 47 siblings"
  survives a traffic spike that fools a week-over-week delta.

## Other signal types (not yet built)

- **Logs**: never grep. Cluster into templates, diff template frequency between
  incident and baseline windows. Return new templates plus rate anomalies.
  Gigabytes become twenty rows.
- **Traces**: exemplars, not dumps. Link slow latency buckets to trace IDs, then
  return a span-level diff against a fast baseline trace.
- **Changes**: query unconditionally on every incident. A large fraction of
  incidents are answered right here, deterministically.
- **Topology**: without the dependency graph the agent can't traverse from a
  frontend symptom to a backend cause — it just describes the symptom in more detail.

## Naming and rollout

Underclaim. Anything called `AutoRCA` sets an expectation it can't meet, and the
first confidently wrong output turns the name into a running joke. Name it for
what it reliably does — gather context, narrow the search, propose hypotheses.

Ship in stages:

1. **Tool layer + auto-generated incident brief.** No reasoning at all. Onset,
   blast radius, correlated changes, top contributing dimensions. Saves on-call
   ten minutes per incident and proves the tool layer.
2. **Hypothesis generation in shadow mode.** Run live, don't show, compare against
   what humans concluded.
3. **Advisory.** Trust earned per incident class.

## Operational constraints

- **Dependency inversion.** The debugger must not depend on the system it debugs.
  Audit the full dependency closure; any edge back into the debugged system is a bug.
- **Orchestrator central, tool execution regional.** Analysis runs next to the
  data; only summaries cross regions. Must be able to run from a region not in
  the incident.
- **Read-only, rate-limited, circuit-broken.** An agent fanning out aggressively
  during a P0 is a real way to turn one incident into two.
- **Trigger from the pager**, target 60–90s to first brief. Output goes to the
  incident channel, not a new dashboard — a separate UI fails on adoption.
- **Permanent replayable record** of every investigation: tool calls, raw results,
  model and skill versions, reasoning trace. Build this on day one; retrofitting
  it after the first bad call is much harder.

## Evaluation

Replay harness over historical incidents with known root causes, against frozen
snapshots. Score root-cause accuracy@1 and @3, time to first correct hypothesis,
tool calls per investigation, and — most importantly — the rate of *confidently
wrong* conclusions. That last number decides whether on-call trusts it.
