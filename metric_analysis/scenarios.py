"""Named incident fixtures.

The single injected rollout that the rest of the suite runs against is a
generous test: one cause, one slice, one onset. Every layer looks competent on
it, and none of them can fail it in an interesting way. These scenarios exist
because a fixture nothing can fail measures the fixture, not the system.

Each one targets a specific way the analysis can be confidently wrong:

  `bad_rollout`         the baseline case, kept identical for replay stability.
  `mix_shift`           nothing degrades; traffic moves toward an already-bad
                        slice. Reporting this as a degradation pages the service
                        owner when the change belongs to whoever owns routing.
  `overlapping_faults`  two unrelated faults at once. Narrowing confidently to
                        one is the failure mode, and the answer on-call needs is
                        "there are two of these".

They are also the arm the skills layer is actually tested by. On a single clean
fault every prior matches and the `confusable_with` discriminators never do any
work; here the wrong prior scores well and the discriminator is the only thing
standing between a plausible story and the right one.
"""

from __future__ import annotations

from .tsdb import Fault, SyntheticTSDB, TrafficShift

__all__ = ["SCENARIOS", "bad_rollout", "mix_shift", "overlapping_faults"]


def bad_rollout(onset: float, seed: int = 11) -> SyntheticTSDB:
    """v2.41 reaches eu-west-4/fb and degrades frontend and txn-coordinator.

    The original fixture, unchanged. Correct output narrows to
    region=eu-west-4, cell=fb with two jobs sharing the blame, driven by rate
    effect rather than mix.
    """
    return SyntheticTSDB(seed=seed, fault_start=onset, rollout_start=onset)


def mix_shift(onset: float, seed: int = 23) -> SyntheticTSDB:
    """A routing change sends traffic to a cell that was always bad.

    us-east-1/ba has run at ~9% errors forever -- known, tolerated, small. At
    `onset` a routing change triples its share, draining the region's other
    cells so the regional total is unchanged.

    Nothing degrades. No error rate moves anywhere. The fleet error rate still
    roughly doubles, purely because more requests now land where they were
    always failing.

    What the analysis must get right:
      * mix effect dominant, rate effect ~0 -- the difference between paging
        the service owner and paging whoever changed routing.
      * fleet-wide *traffic* is flat, because the shift is a redistribution.
        Any check that looks only at fleet totals concludes nothing happened to
        traffic, which is true and completely misleading.
    """
    return SyntheticTSDB(
        seed=seed,
        faults=(
            Fault(
                region="us-east-1",
                cell="ba",
                # Chronic: present in the baseline window too, which is what
                # makes rate_effect vanish and mix_effect carry the change.
                start=None,
                job_error_rates={"frontend": 0.09, "txn-coordinator": 0.09, "tablet-server": 0.09},
                label="chronic-ba",
            ),
        ),
        traffic_shift=TrafficShift(
            start=onset,
            region="us-east-1",
            to_cell="ba",
            from_cells=("aa", "ab", "bb", "fa", "fb"),
            factor=3.0,
        ),
    )


def overlapping_faults(onset_a: float, onset_b: float, seed: int = 31) -> SyntheticTSDB:
    """Two unrelated faults, different regions, different jobs, different times.

    Fault A: eu-west-4/fb frontend, from `onset_a`.
    Fault B: us-east-1/aa tablet-server, from `onset_b`.

    Deliberately disjoint in both region and job, and with no rollout, so no
    single dimension value can explain the change. The correct output refuses
    to narrow to one slice and reports both.

    This is the fixture the min_share guard was written for. It is also where a
    single-onset answer becomes actively misleading: `find_onset` on a fleet
    aggregate returns one timestamp, and whichever fault is larger wins, so the
    other is silently dated wrong.
    """
    return SyntheticTSDB(
        seed=seed,
        faults=(
            Fault(
                region="eu-west-4",
                cell="fb",
                start=onset_a,
                job_error_rates={"frontend": 0.13},
                latency_ms=52.0,
                label="fault-a",
            ),
            Fault(
                region="us-east-1",
                cell="aa",
                start=onset_b,
                job_error_rates={"tablet-server": 0.11},
                lock_ms=36.0,
                label="fault-b",
            ),
        ),
    )


SCENARIOS = {
    "bad_rollout": bad_rollout,
    "mix_shift": mix_shift,
    "overlapping_faults": overlapping_faults,
}
