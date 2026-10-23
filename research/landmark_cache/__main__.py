"""Private native landmark plans, bounded workers, publication and dashboard."""
import argparse
import json
from pathlib import Path

from .plan import make_plan, validate_plan
from .runner import run, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--engine-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile")
    p.add_argument("--output-bucket")
    p.add_argument("--prefix")
    p.add_argument("--offline-sources", type=Path)
    p.add_argument("--local-store", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--wall-seconds", type=int, default=43_200)
    p.add_argument("--memory-gib", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--repair-reason")
    p = sub.add_parser("prefetch")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p = sub.add_parser("publish-local")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--output-bucket", required=True)
    p.add_argument("--prefix", required=True)
    p = sub.add_parser("dashboard")
    p.add_argument("--progress", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.out.exists():
            parser.error("plan output must be new")
        result = make_plan(args.catalog, bucket=args.bucket, engine_sha256=args.engine_sha256,
                           source_commit=args.source_commit)
        write_private(args.out, result)
        print(json.dumps(dict(plan_sha256=result["plan_sha256"], scheduled=len(result["cells"]))))
    elif args.command == "dashboard":
        from .dashboard import render
        render(args.progress, args.out)
    else:
        plan = validate_plan(json.loads(args.plan.read_bytes()))
        if args.command == "prefetch":
            import concurrent.futures
            from .runner import fetch_train
            from .plan import require
            require(1 <= args.workers <= 32 and 0 <= args.shard < args.shards <= 128,
                    "invalid_prefetch_allocation")
            client = s3_client(args)
            assigned = [c for i, c in enumerate(plan["cells"])
                        if i % args.shards == args.shard and c["initial_state"] == "pending"]

            def fetch(cell):
                try:
                    fetch_train(client, cell, args.out)
                    return dict(key=cell["key"], state="verified")
                except Exception as error:
                    return dict(key=cell["key"], state="failed", error_type=type(error).__name__)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                receipts = list(pool.map(fetch, assigned))
            result = dict(format="dope-landmark-TRAIN-prefetch-v1", official_test_access=False,
                          scheduled=len(assigned), verified=sum(r["state"] == "verified" for r in receipts),
                          cells=receipts)
            write_private(args.out / "prefetch.private.json", result)
            print(json.dumps({k: result[k] for k in ("scheduled", "verified")}))
            return
        if args.command == "run" and args.offline_sources:
            from .storage import LocalStore, LocalTrainSource
            if not args.local_store:
                parser.error("offline sources require an explicit local store")
            client, store = LocalTrainSource(args.offline_sources, plan), LocalStore(args.local_store)
        else:
            if not all((args.profile, args.output_bucket, args.prefix)):
                parser.error("S3 transport requires profile, output bucket and prefix")
            client, store = s3_store(args)
        if args.command == "publish-local":
            from .storage import LocalStore
            from .runner import validate_report
            local = LocalStore(args.store)
            published = 0
            for cell in plan["cells"]:
                body = local.lookup(cell)
                if body is not None:
                    report = json.loads(body)
                    validate_report(report, cell["binding"])
                    store.publish(cell, report)
                    published += 1
            print(json.dumps({"published": published, "scheduled": len(plan["cells"])}))
            return
        result = run(plan, binary=args.binary, root=args.out,
                     client=client, store=store,
                     workers=args.workers, threads=args.threads, timeout=args.timeout,
                     memory_bytes=args.memory_gib * 2**30, shard=args.shard, shards=args.shards,
                     wall_seconds=args.wall_seconds, repair_reason=args.repair_reason)
        print(json.dumps({k: result[k] for k in ("state", "scheduled", "terminal", "pending", "counts")}))


def s3_store(args):
    from .storage import S3Store
    client = s3_client(args)
    return client, S3Store(client, args.output_bucket, args.prefix)


def s3_client(args):
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name=args.profile).client("s3", config=Config(
        retries={"total_max_attempts": 1}, connect_timeout=10, read_timeout=60,
        max_pool_connections=getattr(args, "workers", 4) * 2))
    return client


if __name__ == "__main__":
    main()
