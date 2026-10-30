"""The drive kit `qvla realrobot serve` writes beside a run: an optional client for a DROID control stack.

Why: openpi's DROID example client takes three edits and a camera choice for an FR3 run; the kit makes them.
The kit is one directory to copy to the control computer:
- droid_drive.py, that client with the edits made;
- droid_client.py, the request it sends;
- reference_exterior.png, a 320 x 180 frame of your demonstrations' exterior camera (the one the
  converter stored as exterior_image_1_left), so the client finds that camera itself;
- drive.json, the run and the instructions your demonstrations were annotated with, which the client
  offers by number.

The kit's files name no path of this machine, so it runs wherever it is copied (one `scp`).

A re-serve never clobbers a kit file someone edited on disk: each file's bytes are built in memory and
compared with what is already there. Identical, nothing is written. Different, the file is left alone
and printed as kept, unless `replace` (the CLI's `--replace-kit`) asks for this checkout's copy instead,
in which case it is overwritten and printed as replaced.

Offline tier: the frame is decoded with PyAV and written with Pillow. Module scope is stdlib only.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

KIT_FILES = ("droid_drive.py", "droid_client.py")
REFERENCE = "reference_exterior.png"
CAMERA = "observation.images.third_person"      # the training set's name for exterior_image_1_left


def write_drive_kit(lay, rid: str, *, replace: bool = False) -> Path:
    """Write run `rid`'s drive kit (Layout.drive_kit) from this checkout's client files and the
    demonstrations (Layout.demos), and return its directory. The reference frame is the first frame of
    the first task's first episode, at the stored size; the instructions are every task's.

    A kit file already there is kept if it differs from this checkout's (printed, not overwritten)
    unless `replace` is set, in which case it is overwritten and printed as replaced. A file identical
    to what this checkout would write is left untouched either way.

    Raises ValueError, naming the missing input, when the demonstrations' roots.json names no tasks or
    its first task has no `CAMERA` video (an empty corpus or the wrong camera name), instead of letting
    an IndexError or KeyError out as a traceback.
    """
    from PIL import Image

    from axis_qvla.realrobot.fr3.droid_client import frame
    from axis_qvla.realrobot.render.report import episode_frames
    from axis_qvla.signals.methods.cfg_tags import task_prompts

    roots_json = lay.demos / "roots.json"
    doc = json.loads(roots_json.read_text(encoding="utf-8"))
    roots, base = doc.get("roots", doc), lay.demos / doc.get("base", ".")
    tasks = [base / rel for _tid, rel in sorted(roots.items(), key=lambda kv: int(kv[0]))]
    try:
        reference = frame(episode_frames(tasks[0], CAMERA, 0)[0], "exterior")
    except IndexError:
        raise ValueError(f"{roots_json} names no tasks: nothing to take the reference frame from") from None
    except KeyError as exc:
        raise ValueError(f"{tasks[0]} has no {CAMERA!r} camera for the reference frame ({exc})") from None

    prompts = sorted(set().union(*(task_prompts(t) for t in tasks)))
    buf = io.BytesIO()
    Image.fromarray(reference).save(buf, format="PNG")
    fr3 = Path(__file__).resolve().parents[1] / "fr3"
    payloads = {name: (fr3 / name).read_bytes() for name in KIT_FILES}
    payloads[REFERENCE] = buf.getvalue()
    payloads["drive.json"] = (json.dumps({"run": rid, "instructions": prompts, "reference": REFERENCE},
                                          indent=1) + "\n").encode("utf-8")

    kit = lay.drive_kit(rid)
    kit.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        dest = kit / name
        existed = dest.is_file()
        if existed and dest.read_bytes() == data:
            continue
        if existed and not replace:
            print(f"kept {dest} (it differs from this checkout's; pass --replace-kit to replace it)")
            continue
        dest.write_bytes(data)
        if existed:
            print(f"replaced {dest}")
    return kit
