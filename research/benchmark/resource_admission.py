"""Capacity checks including all active study reservations on the host."""

from __future__ import annotations

from .fetch_jope import LIMIT


def assess(host: dict, allocated_scratch_bytes: int,
           active_reservations: list[dict], requested: dict) -> dict:
    """The coordinator must verify reservation liveness before calling this."""
    reservations = [*active_reservations, requested]
    if type(allocated_scratch_bytes) is not int or allocated_scratch_bytes < 0:
        raise ValueError("invalid benchmark scratch allocation")
    blockers = []
    slots = [set(row["cpu_slot"]) for row in reservations]
    if (len(slots) > 4 or any(len(slot) != 16
                            or any(type(core) is not int or not 0 <= core < host["cpu_count"]
                                   for core in slot) for slot in slots)
            or any(slots[i] & slots[j] for i in range(len(slots))
                   for j in range(i))
            or not slots[-1].issubset(host["allowed_cpus"])):
        blockers.append("cpu_reservation_conflict")
    for row in reservations:
        if (type(row["ram_bytes"]) is not int or row["ram_bytes"] <= 0
                or type(row["scratch_bytes"]) is not int or row["scratch_bytes"] <= 0
                or type(row["requires_gpu"]) is not bool):
            raise ValueError("invalid study resource reservation")
    ram = sum(row["ram_bytes"] for row in reservations)
    scratch = sum(row["scratch_bytes"] for row in reservations)
    # Conservatively reserve the full bounds from currently available RAM,
    # including active jobs; their observed usage does not free their reservation.
    if (ram + 2 * 2**30 > host["memory"]["MemTotal"]
            or ram > host["memory"]["MemAvailable"]):
        blockers.append("ram_reservation_capacity")
    if (allocated_scratch_bytes + scratch > LIMIT
            or scratch > host["scratch_disk"]["free_bytes"]):
        blockers.append("scratch_reservation_capacity")
    if sum(row["requires_gpu"] for row in reservations) > 1:
        blockers.append("gpu_reservation_conflict")
    if requested["requires_gpu"]:
        if (host["active_gpu_processes"] or not host["gpus"]
                or host["gpus"][0]["memory_free_mib"] < 17 * 1024):
            blockers.append("gpu_owner_or_vram_capacity")
    if host["load_average"][0] + 16 > 1.25 * host["cpu_count"]:
        blockers.append("cpu_load_capacity")
    return {"admitted": not blockers, "blockers": sorted(blockers),
            "reserved_ram_bytes": ram, "reserved_scratch_bytes": scratch,
            "allocated_scratch_bytes": allocated_scratch_bytes,
            "reserved_cpu_slots": [sorted(slot) for slot in slots],
            "reserved_gpu_jobs": sum(row["requires_gpu"] for row in reservations)}
