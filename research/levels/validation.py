"""Fixed development observers with physical-table counts and no partial-panel means."""
import time
import torch
from .contracts import digest, require
from .data import episode
from .losses import objective


def observe(model, corpus, config, step, *, phase='ordinary', resource_check=None, time_remaining=None):
    config.validate()
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    records, lineage, tasks, families = [], [], {}, {}
    started = time.monotonic()
    observer = dict(format='dope-fixed-development-observer-v2', recipe=config.validation_recipe,
                    scheduled_episodes=config.validation_episodes, state='complete',
                    roles='meta_development_fit_context_tune_query', optimizer_updates=0)
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed+719)
            with torch.no_grad():
                for cursor in range(config.validation_episodes):
                    if time_remaining is not None and not time_remaining():
                        observer['state'] = 'overbudget'; break
                    if resource_check is not None and not resource_check():
                        observer['state'] = 'resource_pause'; break
                    source = corpus.development(cursor, config.validation_recipe)
                    context, query, identity = episode(**{k:source[k] for k in
                        ('rows','kinds','slots','row_ids','task','roles')}, config=model.config,
                        cursor=cursor, seed=corpus.seed)
                    _, raw = objective(model, context, query, source['landmark_input'] if model.config.landmark_input() else source['landmarks'], config, step,
                                       phase=phase, differentiable=False)
                    records.append(raw)
                    family = digest(source['family_id'])
                    families[family] = families.get(family, 0) + 1
                    tasks[source['task']] = tasks.get(source['task'], 0) + 1
                    lineage.append(dict(cursor=cursor, identity=identity, family_sha256=family,
                        physical_table_id=source['source_provenance']['physical_table_id'],
                        context_sha256=digest(context.row_ids), query_sha256=digest(query.row_ids),
                        context_rows=len(context.row_ids), query_rows=len(query.row_ids)))
    finally:
        model.train(was_training)
    observer.update(completed_episodes=len(records), unique_physical_tables=len({
        row['physical_table_id'] for row in lineage}), task_counts=tasks, family_counts=families,
        lineage=lineage, wall_seconds=time.monotonic()-started)
    observer['panel_sha256'] = digest(dict(recipe=config.validation_recipe, episodes=lineage))
    if observer['state'] != 'complete': return None, observer
    if config.validation_recipe == 'owned-balanced-unique-v2':
        require(observer['unique_physical_tables'] == config.validation_episodes
                and tasks == dict(classification=config.validation_episodes//2, regression=config.validation_episodes//2),
                'validation_panel_not_unique_or_balanced')
    means = {level:{key:sum(r[level][key] for r in records)/len(records)
                    for key in ('total','generation','numeric','categorical','missing','transfer_proxy',
                                'driver_proxy','profile','consistency')}
             for level in records[0]}
    return means, observer
