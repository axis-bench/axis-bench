"""The FR3 real-robot line: embodiment and line declarations, norm statistics, and the checks that keep
the declared training and serving consistent (serve geometry, state order, action units).

The line's nine run specs (experiments/realrobot/runs/) are declarations that name their inputs `local:`;
`fr3 local-runs` writes the ones `qvla train` trains. See docs/tutorials/real-world-your-workspace.md.
Offline tier.
"""
