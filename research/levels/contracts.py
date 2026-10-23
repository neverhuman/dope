"""Versioned, bounded research contracts with explicit family and data roles."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

LEVELS = {"low": 256, "medium": 1024, "high": 4096, "ultra": 5120}
BLOCKS = (256, 768, 3072, 1024)
LAYOUT = "nested-256-1024-4096-plus-1024-v1"
PREPROCESSING = "fit-only-positional-mixed-v1"


class ResearchError(ValueError):
    """Static, source-free diagnosis and repair instructions."""
    purpose = "prepare and execute a bounded shared-table-model experiment"
    reason = "input, model or experiment identity violates its declared contract"
    common_fixes = "check the pinned manifest, shape limits, source roles and checkpoint recipe"
    repair_hint = "preserve the failed run; repair its declared inputs before starting a new attempt"
    docs_url = "research/levels/README.md"


def require(value, code):
    if not value:
        raise ResearchError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def levels(value):
    aliases = {"small": "low", "med": "medium", "large": "high"}
    selected = ["low", "medium", "high"] if value == "all" else value.split(",")
    selected = [aliases.get(s.strip(), s.strip()) for s in selected]
    require(selected and len(set(selected)) == len(selected)
            and all(s in LEVELS for s in selected), "invalid_level_selection")
    return sorted(selected, key=LEVELS.get)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    heads: int = 8
    feature_blocks: int = 4
    inducing_blocks: int = 2
    inducing_slots: int = 64
    summary_slots: int = 32
    target_blocks: int = 2
    decoder_blocks: int = 6
    condition_slots: int = 8
    max_features: int = 64
    max_categories: int = 64
    max_context: int = 1024
    max_query: int = 128
    cell_cap: int = 65536
    diffusion_steps: int = 32
    layout: str = LAYOUT
    graph: str = "mixed-code-diffusion-v1"

    def validate(self):
        require(self.layout == LAYOUT and self.graph == "mixed-code-diffusion-v1", "unknown_model_graph")
        for name in ("width", "heads", "feature_blocks", "inducing_blocks", "inducing_slots",
                     "summary_slots", "target_blocks", "decoder_blocks", "condition_slots",
                     "max_features", "max_categories", "max_context", "max_query", "cell_cap",
                     "diffusion_steps"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, "invalid_model_dimension")
        require(self.width <= 512 and self.width % self.heads == 0 and self.heads <= 16
                and self.feature_blocks <= 8 and self.inducing_blocks <= 4
                and self.target_blocks <= 4 and self.decoder_blocks <= 12
                and self.inducing_slots <= 128 and self.summary_slots <= 64
                and self.condition_slots <= 16 and self.max_features <= 256
                and 2 <= self.max_categories <= 256 and self.max_context <= 16384
                and self.max_query <= 1024 and self.cell_cap <= 262144
                and self.diffusion_steps <= 64, "model_envelope_exceeded")
        return self

    def manifest(self):
        return asdict(self.validate())


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    seed: int = 20261023
    accumulation: int = 64
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    warmup_fraction: float = .02
    weight_decay: float = .01
    gradient_clip: float = 1.
    validate_every: int = 50
    checkpoint_every: int = 50
    proxy_rows: int = 32
    transfer_weight: float = .25
    driver_weight: float = .10
    profile_weight: float = .10
    consistency_weight: float = .05
    proxy_ramp_steps: int = 100
    precision: str = "fp32"
    wall_seconds: int = 600
    cpu_threads: int = 4
    device: str = "cpu"

    def validate(self):
        require(all(type(v)is int for v in (self.steps,self.seed,self.accumulation,self.validate_every,
                self.checkpoint_every,self.proxy_rows,self.proxy_ramp_steps,self.wall_seconds,self.cpu_threads))
                and 0<=self.seed<2**32, 'training_integer_required')
        require(1 <= self.steps <= 10000000 and 1 <= self.accumulation <= 256
                and 1 <= self.validate_every <= self.steps
                and 1 <= self.checkpoint_every <= self.steps
                and 4 <= self.proxy_rows <= 128 and self.proxy_ramp_steps >= 0
                and 1 <= self.wall_seconds <= 7200 and 1 <= self.cpu_threads <= 16
                and self.precision in ("fp32", "bf16")
                and self.device in ("cpu", "cuda"), "invalid_training_allocation")
        require(all(math.isfinite(v) and v >= 0 for v in (
            self.learning_rate, self.min_learning_rate, self.warmup_fraction,
            self.weight_decay, self.gradient_clip, self.transfer_weight,
            self.driver_weight, self.profile_weight, self.consistency_weight))
                and 0 < self.min_learning_rate <= self.learning_rate
                and 0 <= self.warmup_fraction < 1 and self.gradient_clip > 0,
                "invalid_optimizer_recipe")
        return self


def validate_corpus(manifest):
    """No unresolved or cross-role families; only owned TRAIN exports enter real episodes."""
    require(manifest.get("format") == "dope-meta-corpus-v1", "unknown_corpus")
    families, identities = {}, set()
    for item in manifest["datasets"]:
        family, role = item["family_id"], item["meta_role"]
        require(role in ("train", "development", "final") and family
                and not family.startswith("unresolved:"), "unresolved_source_family")
        require(families.setdefault(family, role) == role, "family_role_overlap")
        require(item["source_role"] == "official_train" and item["rights"]["admitted"] is True
                and bool(item["rights"]["license_id"]), "source_not_admitted")
        identity = item["train_sha256"]
        require(len(identity) == 64 and all(c in "0123456789abcdef" for c in identity), "invalid_source_digest")
        require(identity not in identities, "duplicate_source_identity")
        identities.add(identity)
        require(type(item["target_index"]) is int and item["target_index"] >= 0,
                "target_declaration_required")
        require(item["split_policy"] in ("iid", "group", "time"), "unknown_row_policy")
    require(any(r == "train" for r in families.values())
            and any(r == "development" for r in families.values()), "missing_meta_roles")
    return manifest


def validate_schema(schema, config=None):
    config=ModelConfig() if config is None else config
    require(set(schema)=={'preprocessing','task','target_index','output','columns'}
            and schema['preprocessing']==PREPROCESSING
            and schema['output']=='positional_numeric_and_category_ids', 'unknown_public_schema')
    columns=schema['columns'];target=schema['target_index']
    require(isinstance(columns,list) and 1<=len(columns)<=config.max_features+1,'schema_shape_unsupported')
    seen=set()
    for i,column in enumerate(columns):
        require(set(column)=={'slot','kind','center','scale','categories'},'unexpected_schema_metadata')
        slot=column['slot']
        require(type(slot)is int and 0<=slot<2**32 and slot not in seen
                and (slot==0)==(target==i),'schema_binding_invalid')
        seen.add(slot)
        require(type(column['center'])in(int,float) and type(column['scale'])in(int,float)
                and math.isfinite(column['center']) and math.isfinite(column['scale'])
                and column['scale']>0 and type(column['categories'])is int,'schema_transform_invalid')
        require((column['kind']=='numeric' and column['categories']==0)
                or (column['kind']=='categorical' and 3<=column['categories']<=config.max_categories
                    and column['center']==0 and column['scale']==1),'schema_kind_unsupported')
    if target is None:
        require(schema['task']is None and len(columns)<=config.max_features,'target_free_schema_invalid')
    else:
        require(type(target)is int and target==len(columns)-1
                and (schema['task'],columns[-1]['kind'])in
                    (('regression','numeric'),('classification','categorical')),'target_schema_invalid')
    return schema
