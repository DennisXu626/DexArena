"""Record-neutral fingerprint transitions of the released environments.

A dataset's certificate (``meta/recording_integrity.json``) and the identity in its
replay sidecar (``replay/source_*/meta.json``) carry the per-environment code
fingerprint (``tasks.config.code_fingerprint``) of the build that recorded it. Some
later edits move that fingerprint without changing anything the environment records,
for example a fix that only acts on a simulation backend the data was not recorded
with. Such an edit is registered here as ``(old, new, evidence)`` once it has been
measured record-neutral, and every tool that compares a recorded fingerprint with the
current build resolves both through :func:`resolve` first. The dataset files keep the
fingerprint they were written with.

An edit that changes what the environment records is a new version, not a lineage
entry: data recorded under the old fingerprint stays a separate dataset and keeps
reporting ``fingerprint DRIFT``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

_GPU_STAGE_BODY_WEIGHT = (
    "tasks/cabinet_pouring/cabinet_pouring_long.py: _apply_gpu_stage_body_weight added to "
    "_before_simulation_step. On physx_cuda, set_disable_gravity(False) changes the flag but "
    "not the simulation, so the detached cap twin and the open bottle were weightless and, with "
    "zero linear damping, drifted at constant velocity; the method applies m*g to them on every "
    "substep on the GPU backend only and returns before reading or writing anything on the CPU "
    "backend the datasets were recorded with. Record-neutral: recordings through the "
    "teleoperation entry point (fixed seed) before and after the edit are byte-identical "
    "(episode.npz, raw_states.pkl, env_states.pkl) on all five targets of the family; CPU "
    "action replay of three UnscrewCap episodes that cross attached cap -> free twin -> placed "
    "-> open bottle gives an identical per-step state hash chain before and after (3/3); "
    "replay_success_rate over the four released datasets of the family (OpenCabinet, "
    "PickBottle, PourBeads, UnscrewCap; 100 episodes each) gives identical per-episode rows "
    "(steps, success verdicts, first divergence, final error) before and after, 400/400."
)

_CREATION_TIME_MATERIALS = (
    "Contact materials of the articulated task objects (package carton, office "
    "laptop/mouse/keyboard, threaded nut, screw bottle and cap, lunch box, refrigerator, "
    "blender) are bound while the URDF loader or articulation builder creates the collision "
    "shapes instead of being assigned to the built links with set_physical_material. Same "
    "material objects and values; the per-shape material read-back is identical before and "
    "after (10 environments). On physx_cuda a material assigned to articulation links after the "
    "build leaves their contacts without friction (the PackageOpening apple glided over the "
    "carton floor at a constant 3.9 cm/s and bounced off the walls; with the material bound at "
    "creation it rests and a 5 cm/s kick stops within one control step). The CPU pipeline is "
    "unchanged: headless_drive_all through the real teleop entry point with a fixed seed "
    "recorded byte-identical episode.npz, raw_states.pkl and env_states.pkl before and after "
    "for all 26 affected environments (78/78 files), and a same-tree repeat is identical "
    "(determinism control)."
)

#: env_id -> ((old_fingerprint, new_fingerprint, evidence), ...), 12-hex prefixes.
LINEAGE: Mapping[str, Sequence[tuple[str, str, str]]] = {
    "CabinetPouringLong-Sharpa-v1": (
        ("75ebc737dcbb", "48d4ba761ad7", _GPU_STAGE_BODY_WEIGHT),
        ("48d4ba761ad7", "2fbe02e661cf", _CREATION_TIME_MATERIALS),
    ),
    "CabinetPouringOpenCabinet-Sharpa-v1": (
        ("2accab13a715", "141205a29a6e", _GPU_STAGE_BODY_WEIGHT),
        ("141205a29a6e", "b32a7a3c13a7", _CREATION_TIME_MATERIALS),
    ),
    "CabinetPouringPickBottle-Sharpa-v1": (
        ("8a4a1e792709", "a7853bbc4f11", _GPU_STAGE_BODY_WEIGHT),
        ("a7853bbc4f11", "125a59a94882", _CREATION_TIME_MATERIALS),
    ),
    "CabinetPouringPourBeads-Sharpa-v1": (
        ("5071b207bc01", "b2122d450589", _GPU_STAGE_BODY_WEIGHT),
        ("b2122d450589", "761222b38034", _CREATION_TIME_MATERIALS),
    ),
    "CabinetPouringUnscrewCap-Sharpa-v1": (
        ("827f7215c131", "e7828140b26f", _GPU_STAGE_BODY_WEIGHT),
        ("e7828140b26f", "5518f071253a", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationAppleToBlender-Sharpa-v1": (
        ("f6c8793af0a4", "9bb33af809a4", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationCloseFridge-Sharpa-v1": (
        ("64a508d14927", "6c6b8be47e0d", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationLong-Sharpa-v1": (
        ("d445da53c3aa", "1359e6532583", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationOpenFridge-Sharpa-v1": (
        ("0e2b69ed8c51", "c67cda1745e5", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationPlaceLid-Sharpa-v1": (
        ("301f35d6d888", "a1f9de628a13", _CREATION_TIME_MATERIALS),
    ),
    "JuicePreparationPressButton-Sharpa-v1": (
        ("7338a38588f1", "4b5562e17e78", _CREATION_TIME_MATERIALS),
    ),
    "MealPreparationLong-Sharpa-v1": (
        ("a07b43adca2f", "75bc7a38fe42", _CREATION_TIME_MATERIALS),
    ),
    "MealPreparationOpenLunchBox-Sharpa-v1": (
        ("9a134a95c95e", "131de364f39c", _CREATION_TIME_MATERIALS),
    ),
    "MealPreparationPlaceBread-Sharpa-v1": (
        ("f787f71f6728", "aa8ccfe0b596", _CREATION_TIME_MATERIALS),
    ),
    "MealPreparationScoopTomato-Sharpa-v1": (
        ("e3d73d3a6179", "3599f2537443", _CREATION_TIME_MATERIALS),
    ),
    "NutBoltLong-Sharpa-v1": (
        ("0659e8c9e44e", "f83dd301400e", _CREATION_TIME_MATERIALS),
    ),
    "NutBoltPickBolt-Sharpa-v1": (
        ("d9f90cf86685", "8796da5e4b50", _CREATION_TIME_MATERIALS),
    ),
    "NutBoltSeatNut-Sharpa-v1": (
        ("9cbaf3acf05a", "1112e0ebe501", _CREATION_TIME_MATERIALS),
    ),
    "NutBoltTurnNut-Sharpa-v1": (
        ("1307e90d9776", "be26aa60a9cb", _CREATION_TIME_MATERIALS),
    ),
    "OfficeDeskClickMouse-Sharpa-v1": (
        ("2160f6292a77", "7fbc174297d7", _CREATION_TIME_MATERIALS),
    ),
    "OfficeDeskLong-Sharpa-v1": (
        ("2c339f94c682", "fbc516728c2c", _CREATION_TIME_MATERIALS),
    ),
    "OfficeDeskOpenLaptop-Sharpa-v1": (
        ("6d55b5401baf", "e759b2ecdbbc", _CREATION_TIME_MATERIALS),
    ),
    "OfficeDeskTypeKeys-Sharpa-v1": (
        ("788afde2f712", "02bda1de2870", _CREATION_TIME_MATERIALS),
    ),
    "PackageOpeningExtractApple-Sharpa-v1": (
        ("bd112c9cd4c6", "6527e485dcaa", _CREATION_TIME_MATERIALS),
    ),
    "PackageOpeningLong-Sharpa-v1": (
        ("b7b50ebdeb3e", "a22acc10e678", _CREATION_TIME_MATERIALS),
    ),
    "PackageOpeningOpenPackage-Sharpa-v1": (
        ("9f5110b2e174", "2e0ac84514f1", _CREATION_TIME_MATERIALS),
    ),
}


def _bare(version: str) -> str:
    return version[3:] if version.startswith("fp:") else version


def resolve(env_id: str, version: str) -> str:
    """The head of ``version``'s chain for ``env_id``; unchanged when no entry applies.

    Accepts and returns either the bare 12-hex form or the certificate's ``fp:`` form.
    """
    prefix = "fp:" if version.startswith("fp:") else ""
    current = _bare(version)[:12]
    chain = {old: new for old, new, _evidence in LINEAGE.get(env_id, ())}
    seen = set()
    while current in chain and current not in seen:
        seen.add(current)
        current = chain[current]
    return prefix + current


def equivalent(env_id: str, a: str, b: str) -> bool:
    """Whether two fingerprints of ``env_id`` resolve to the same head."""
    return resolve(env_id, a)[-12:] == resolve(env_id, b)[-12:]
