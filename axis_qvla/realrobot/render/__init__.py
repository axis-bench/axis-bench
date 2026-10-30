"""Kinematic REALALIGN-style renderer: recorded AXIS attempts -> a Franka + Robotiq 2F-85 LeRobot v3 corpus.

RENDER TIER: mujoco, mink, PyAV, numpy, scipy, pandas, pyarrow -- one venv, created from
axis_qvla/realrobot/render/render.lock.txt. Never import jax, torch or lerobot here, and never import this
package at module scope from axis_qvla.realrobot.realign, .realdata or .fr3 (offline tier); the FR3 checks
read the stdlib-only `rig` inside the functions that need it.

One attempt: source (packed MuJoCo state + scene) -> assemble (the recorded scene, its hand swapped
for a 2F-85, on the table) -> ik (TCP-space mink IK at the recorded rows) -> episode (15 fps states,
ZOH actions) -> draw (two cameras) -> writer (LeRobot v3 + a per-episode sidecar). Then droid8d (the
training view), labels (curve store) and gates. Kinematic only: objects follow their recorded poses
and nothing is simulated. See docs/tutorials/real-world-your-workspace.md.
"""
