"""Validate a catalog config. Run: python3 -m metric_analysis.check catalog/

Exists so a catalog edit can be checked before it reaches an incident. A config
that fails to load during a P0 is the worst possible time to find out.
"""

from __future__ import annotations

import sys

from .config import CatalogError, describe_catalog, load_catalog


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        catalog = load_catalog(argv[1])
    except CatalogError as e:
        print(f"INVALID  {e}", file=sys.stderr)
        return 1
    print(describe_catalog(catalog))
    signals = catalog.golden_signals()
    missing = [s for s in catalog.signal_order if s not in signals]
    if missing:
        # Not fatal, but the sweep will simply have nothing to report for these,
        # and a signal silently absent looks the same as a signal that is healthy.
        print(f"\nWARNING: no metric is tagged for signal(s): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
