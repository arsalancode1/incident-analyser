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

| Layer | What it is | Change rate | State |
|---|---|---|---|
| Harness | Loop control, dispatch, validation, budgets, evidence ledger, replay/eval | Rarely | `incident_agent/`, minus the eval half |
| Tools | Data access and analysis | With data sources | `metric_analysis/`; metrics only |
| Skills | Domain semantics, incident-class priors, heuristics | Constantly | `skills/*.json` |
| Agent | Orchestrator prompt, sub-agent prompts, output schema | Rarely, eval-gated | `incident_agent/prompts.py` |

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
- **Prefer structure over instruction.** Anything a prompt asks for politely can
  be declined politely. Concluding is a schema-validated tool call, so the
  falsification test is a required field and a hypothesis citing an unknown
  evidence id is rejected — mandatory falsification enforced by the type system.
  Windows are named (`incident`, `baseline`, `scan`) rather than numeric,
  because a model asked for timestamps invents plausible ones and an
  investigation anchored on a hallucinated window produces real-looking analysis
  of the wrong hour.
- **Establish that something happened before explaining it.** Contribution
  analysis decomposes whatever delta it is given, and explanatory power is a
  *share* of that delta — so on a flat metric some slice still owns 70% of the
  noise and gets named with a straight face. Attribution runs only after a
  change point, a delta past a floor, or a peer outlier establishes the change
  is real. This was not theoretical: the first brief confidently located an
  incident in a completely healthy fleet.

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
- **Sweep every golden signal, not only the one that paged.** The paged metric
  is one view of the service. Errors can be flat while traffic collapses, and a
  brief anchored on the page would report "errors normal" and miss the outage.
  Each signal is judged on its own terms rather than for correlation with the
  symptom, so an independent problem stays visible. Signals are configured, not
  hardcoded: the four classic names are a default, and a queueing system
  reasonably adds `queue_depth`.
- **Rate versus mix is not partition-invariant, so compute it once on the joint
  partition.** Cell names repeat across regions by design, so grouping by cell
  alone pools a cell gaining traffic with its healthy namesakes; the composition
  inside that pooled slice changes and a pure traffic shift reappears as a rate
  effect. Computing the mechanism from whichever field greedy narrowing happened
  to pick first also makes the answer depend on a search order that carries no
  meaning.
- **Fleet aggregates hide the incidents that matter most.** A redistribution
  preserves the total, so fleet-wide traffic reads flat while the error rate
  doubles. Reporting fleet numbers is correct and, on its own, misleading —
  which is why blast radius is a peer comparison rather than a total.

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

## Decisions made while building

Recorded here so they are not relitigated, and because each was a bug first.

- **Materiality gate before attribution.** See above. The failure was silent and
  confident, which is the combination that costs trust.
- **The narrowing floor is absolute, not relative.** Comparing each slice
  against a fraction of the largest one puts the bar at whatever height the
  biggest contributor happens to land on: an 0.81/0.19 split dropped a genuinely
  broken slice while 0.70/0.29 kept one no more material. Materiality does not
  depend on how big the largest contributor was.
- **Skills rank on distinguishing evidence, not generic evidence.** "Symptom is
  errors, errors are material, onset is a step" describes almost every error
  incident. Weighting those highly made a bad-rollout prior rank first on a
  traffic-shift incident with no rollout. Generic criteria now establish only
  relevance; specific ones rank. A distinguishing criterion that was *checkable
  and came back absent* counts against a skill — but only where the evidence
  exists, since penalising on unobserved data is the absence-of-evidence trap.
- **Model access is provider-agnostic, behind a one-method protocol.** Not
  neutrality for its own sake: comparing models on the replay harness is the
  only way to learn whether the expensive one is better at *sequencing*, which
  is the only job the model has, and that is unanswerable if a vendor is welded
  into the loop. It also makes the loop testable offline — every harness test
  runs against a scripted client with no network — and keeps SDKs out of the
  dependency set, since they are imported lazily.
- **The verifier is denied the reasoning trace.** A coherent chain of thought is
  exactly what makes an unsupported conclusion persuasive, so a verifier that
  reads it will be talked round by it. It strips claims rather than softening
  them: a hedged wrong claim still misdirects and is harder to notice.
- **Fixtures must be failable.** A single clean fault makes every layer look
  competent and cannot distinguish a system that reasons from one that got
  lucky. The mix-shift and overlapping-fault scenarios exist to be failed, and
  both found real bugs within minutes of existing.

## Wiring a real backend

The tool layer assumes things about storage it cannot check per query, and a
backend that quietly violates one produces confident, well-formatted, wrong
output. So conformance is a suite you run once, not a hope.

The load-bearing assumption is invariant 1: a filter matching no entity must
raise, not return zero series. Storage has no reason to distinguish these on its
own — an unmatched filter and a silent entity are both "no data" — so this is
the single most likely thing to be silently wrong in an integration. Where the
catalog enumerates a field's values it can enforce this itself; where values are
resolved live, which any high-cardinality field requires, enforcement moves down
to the backend and the contract becomes load-bearing rather than belt-and-braces.

For Monarch specifically: alignment must match the metric kind (counters delta,
gauges mean) or window sums become resolution-dependent and the two-pass onset
detection compares two different universes. And Monarch stores latency as
mergeable bucketed distributions, which means the "percentiles aren't additive"
gap is an artifact of pre-computed quantiles rather than a law — real bucket
merging would make latency attribution tractable.

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

Every investigation already emits a JSON record — brief, priors with versions,
every tool call and result, transcript, conclusion, verification — because
retrofitting that after the first bad call is much harder. The replay harness
consumes it; it is not built yet.

Replay harness over historical incidents with known root causes, against frozen
snapshots. Score root-cause accuracy@1 and @3, time to first correct hypothesis,
tool calls per investigation, and — most importantly — the rate of *confidently
wrong* conclusions. That last number decides whether on-call trusts it.
