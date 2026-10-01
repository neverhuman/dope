"""Pin the author synthpop CART source, rights, runtime, and pilot objective."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .score import sha256


HERE = Path(__file__).parent
SCRATCH = Path("/mnt/fast-scratch/dope-benchmark")
ARCHIVE = SCRATCH / "source-snapshots/synthpop_1.9-3.tar.gz"
ENV_LOCK = SCRATCH / "envs/synthpop-r44.explicit.txt"
AUDIT = HERE / "synthpop-cart-source.lock.json"
SOURCE_MEMBERS = ("DESCRIPTION", "R/syn.r", "R/sampler.syn.r",
                  "R/functions.syn.r", "R/utility.syn.r")
GRID = [
    {"minbucket": 5, "cp": 1e-8},
    {"minbucket": 10, "cp": 1e-8},
    {"minbucket": 20, "cp": 1e-8},
    {"minbucket": 10, "cp": 1e-5},
]


def source_members() -> tuple[dict[str, str], str]:
    hashes = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in SOURCE_MEMBERS:
            member = archive.getmember("synthpop/" + name)
            stream = archive.extractfile(member)
            if stream is None or not member.isfile():
                raise ValueError("synthpop author source member is missing")
            data = stream.read()
            hashes[name] = hashlib.sha256(data).hexdigest()
            if name == "DESCRIPTION":
                description = data.decode()
    if "Version: 1.9-3" not in description \
            or "License: GPL-2 | GPL-3" not in description \
            or "Repository: CRAN" not in description:
        raise ValueError("synthpop version or source rights changed")
    return hashes, description


def build() -> dict:
    members, _ = source_members()
    if (sha256(ARCHIVE) !=
            "65d4146e8ca5ac1236d42a89eec057291f57b1224645e958909f9539b9c54ff5"
            or sha256(ENV_LOCK) !=
            "a96d501cf4fa21609c96e85fd61713131c2a3caa879cb0055e10030a8a77def7"):
        raise ValueError("synthpop source archive or pinned runtime changed")
    return {
        "format": "dope-synthpop-cart-source-audit", "version": 1,
        "package": "synthpop", "package_version": "1.9-3",
        "upstream_url": "https://cran.r-project.org/package=synthpop",
        "source_archive_sha256": sha256(ARCHIVE),
        "source_member_sha256": members,
        "source_license": "GPL-2 | GPL-3",
        "adapter_license": "GPL-2.0-only OR GPL-3.0-only",
        "adapter_role": "restricted_research_only_not_mit_product_runtime",
        "runtime_explicit_lock_sha256": sha256(ENV_LOCK),
        "adapter_source_sha256": sha256(HERE / "synthpop_cart.R"),
        "contract_probe_sha256": sha256(HERE / "tests/probe_synthpop_cart.R"),
        "auditor_source_sha256": sha256(Path(__file__)),
        "representation": "common_numeric_projection_only",
        "fit_api": "synthpop::syn(data,method='cart',models=TRUE)",
        "sample_api": "restricted_GPL_adapter_from_fitted_rpart_leaf_donors",
        "default_config": GRID[0], "tuning_search_space": GRID,
        "tuning_trials_per_dataset": 4,
        "total_trial_cap_per_dataset": 8,
        "total_wall_cap_seconds_per_dataset": 43_200,
        "native_objective": {
            "name": "validation_cart_pMSE", "direction": "minimize",
            "api": "synthpop::utility.gen(synthetic,validation,method='cart',resamp.method='none')$pMSE",
            "implementation_source_sha256": members["R/utility.syn.r"],
            "tie_breaks": ["artifact_bytes_ascending", "config_sha256_ascending"],
        },
        "artifact_policy": "restricted_donor_values_and_fitted_rpart_models_charged_in_full",
        "formal_dp_claim": False,
        "release_l3_claim": False,
        "official_tests_opened": False,
        "mfs_v2": None, "ptf_v1": None,
        "matrix_admission": "pending_methods_lock_update_after_active_ARF_rounds",
    }


def main() -> None:
    report = build()
    AUDIT.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"source_archive_sha256": report["source_archive_sha256"],
                      "adapter_source_sha256": report["adapter_source_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
