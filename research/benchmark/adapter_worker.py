"""Isolated fit/sample process so native-code stalls obey the job deadline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import adapters


def main() -> None:
    request = json.load(sys.stdin)
    try:
        binary = Path(request["binary"]) if request.get("binary") else None
        if request["method"] in ("CTGAN", "TVAE"):
            from . import sdv_adapter
            if request["action"] == "fit":
                if request["metadata"]["task"] != "regression":
                    raise ValueError("SDV adapter is currently audited for regression only")
                config = {key: value for key, value in request["config"].items()
                          if key not in ("fit_timeout_seconds", "sample_timeout_seconds")}
                result = {"status": "ok", **sdv_adapter.fit(
                    request["method"], Path(request["train"]), config,
                    request["seed"], Path(request["artifact_dir"]))}
            elif request["action"] == "sample":
                result = {"status": "ok", **sdv_adapter.sample(
                    Path(request["artifact_dir"]), request["row_count"], request["seed"],
                    Path(request["output"]))}
            else:
                raise ValueError("unknown adapter action")
            print(json.dumps(result, sort_keys=True))
            return
        if request["action"] == "fit":
            files = adapters.fit(request["method"], Path(request["train"]),
                                 request["metadata"], request["config"],
                                 request["seed"], Path(request["artifact_dir"]), binary)
            result = {"status": "ok", "files": files}
        elif request["action"] == "sample":
            adapters.sample(request["method"], Path(request["artifact_dir"]),
                            request["row_count"], request["seed"],
                            Path(request["output"]), binary)
            result = {"status": "ok"}
        else:
            raise ValueError("unknown adapter action")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, TimeoutError) as error:
        result = {"status": "failed", "error_type": type(error).__name__}
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
