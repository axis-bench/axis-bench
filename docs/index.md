# AXIS-Bench

AXIS-Bench measures how quality labels in pretraining data change a vision-language-action (VLA) policy
after fine-tuning. Each VLA (π0.5, SmolVLA, X-VLA) is pretrained on the same 5,000 simulated Franka
demonstrations at the same budget, fine-tuned on the same demonstrations of 24 held-out tasks, and evaluated
on 50 seeds per task. The real-world line re-renders 19,250 simulation demonstrations for your workspace.

![AXIS-Bench overview: human demonstrations, state-derived quality, quality labels, quality-aware pretraining of three VLAs, the simulation benchmark and re-rendering for a real workspace](assets/overview.png)

Every path starts with [Install](install.md).

| you want to | you need | start with |
|---|---|---|
| reproduce the paper's tables from the published results | a CPU | [From published results](tutorials/reproduce-from-results.md) |
| re-evaluate the released checkpoints | a GPU with EGL | [Re-evaluate released checkpoints](tutorials/reeval-released-checkpoints.md) |
| train with your own quality method | GPUs | [Write a quality method](tutorials/write-a-quality-method.md) |
| evaluate your own model | a GPU with EGL, and what your model needs | [Connect your own model](tutorials/connect-your-own-model.md) |
| run it on a real robot | CPUs to render, GPUs to train, a Franka with a Robotiq 2F-85 gripper on DROID | [Real robot](tutorials/real-world-your-workspace.md) |

[Paper](https://arxiv.org/abs/2610.18869) · [Website](https://axis-bench.github.io/) ·
[Code](https://github.com/axis-bench/axis-bench) · [Data](https://huggingface.co/axisrobotics)
