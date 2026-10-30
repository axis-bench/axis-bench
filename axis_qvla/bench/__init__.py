"""The benchmark, as a package: what is scored, on which seeds and layouts, in which conventions.

Everything a benchmark user touches goes through here. `axis_qvla.bench.spec` loads the tracked
declaration under conf/benchmarks/axis-heldout-v1/ and is the ONLY place that knows its file layout; the eval
harness, the campaign driver and the table generator ask it rather than re-declaring task lists.

Core tier: stdlib only, so any interpreter can import it: both venvs and the CLI's `bench`
commands.
"""
