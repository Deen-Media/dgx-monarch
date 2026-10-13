"""Fleet job assembly: one job dict per conditioning entry or prompt line."""
from __future__ import annotations

MAX_FLEET_JOBS = 1024


def build_jobs(prompts_text: str, positive_list, negative_list, base_seed: int):
    """Assemble one job dict (positive, text, negative, seed) per job.

    Conditioning-list jobs come first, then prompt lines, which carry their raw
    ``text`` and a None ``positive`` until the caller encodes them.
    """
    jobs = []
    for cond in positive_list or []:
        jobs.append({"positive": cond, "text": None})
        if len(jobs) > MAX_FLEET_JOBS:
            raise ValueError(
                f"fleet: more than {MAX_FLEET_JOBS} jobs exceeds the per-prompt limit")
    for line in (prompts_text or "").splitlines():
        line = line.strip()
        if line:
            jobs.append({"positive": None, "text": line})
            if len(jobs) > MAX_FLEET_JOBS:
                raise ValueError(
                    f"fleet: more than {MAX_FLEET_JOBS} jobs exceeds the per-prompt limit")
    if len(jobs) > MAX_FLEET_JOBS:
        raise ValueError(
            f"fleet: {len(jobs)} jobs exceeds the per-prompt limit of {MAX_FLEET_JOBS}")
    negatives = list(negative_list or [])
    if len(negatives) > 1 and len(negatives) != len(jobs):
        raise ValueError(
            f"negative list has {len(negatives)} entries but there are {len(jobs)} "
            "jobs; provide one negative (broadcast) or exactly one per job")
    for i, job in enumerate(jobs):
        job["negative"] = negatives[i] if len(negatives) > 1 else (
            negatives[0] if negatives else None)
        job["seed"] = int(base_seed) + i
    return jobs
