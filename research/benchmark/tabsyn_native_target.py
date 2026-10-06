"""TRAIN-fitted target inverse only; common projection buffers stay unchanged."""
import hashlib
import json
import math

TRACK = 'author KPI on TRAIN-projection-clipped source target units'


def require(ok, reason):
    if not ok: raise ValueError(reason)


def target_state(projection_bytes, expected_sha256):
    require(type(projection_bytes) is bytes and type(expected_sha256) is str
            and len(expected_sha256) == 64 and all(c in '0123456789abcdef' for c in expected_sha256),
            'native_projection_binding_invalid')
    require(hashlib.sha256(projection_bytes).hexdigest() == expected_sha256, 'native_projection_hash_changed')
    p = json.loads(projection_bytes)
    require(type(p.get('version')) is int and p['version'] == 1 and p.get('task') == 'regression'
            and p.get('fit_partition') == 'train' and p.get('numeric_rule') == 'train_minmax_clip_0_1',
            'native_projection_not_frozen_TRAIN_regression')
    limits = p['target_map']
    require(type(limits) is dict and set(limits) == {'min', 'max'}
            and all(type(v) in (int, float) and math.isfinite(v) for v in limits.values())
            and limits['min'] <= limits['max'] and math.isfinite(limits['max'] - limits['min']),
            'native_TRAIN_target_state_invalid')
    # These private values never appear in the public native receipt.
    return dict(limits)


def inverse_values(projected_values, state):
    require(type(projected_values) in (list, tuple)
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in projected_values),
            'native_projected_target_invalid')
    result = [state['min'] + v * (state['max'] - state['min']) for v in projected_values]
    require(all(math.isfinite(v) for v in result), 'native_inverse_target_nonfinite')
    return result


def inverse_matrix_targets(matrix, target_index, state):
    require(type(target_index) is int and 0 <= target_index < matrix.shape[1], 'native_target_index_invalid')
    result = matrix.copy()
    result[:, target_index] = inverse_values(matrix[:, target_index].tolist(), state)
    return result


def uninformative_receipt(reason):
    return dict(status='tuning_inapplicable_uninformative_author_target', reason=reason,
        native_value=None, default_retained=True, native_winner=None, native_track=TRACK,
        outside_fit_extrema_reconstructed=False, common_projection_units_changed=False)


def generated_author_labels(source_values):
    """Generated stdlib controls only; the admitted evaluator uses exact NumPy float32 labels."""
    return [math.log(max(1, min(20000, v))) for v in source_values]


def check_validation_partition(provenance, worker_sha256, projection_sha256):
    require(all(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in (worker_sha256, projection_sha256)), 'native_input_identity_invalid')
    require(type(provenance) is dict
            and provenance.get('partition') == 'official_training_derived_validation'
            and provenance.get('official_tests_opened') is False
            and provenance.get('worker_sha256') == worker_sha256
            and provenance.get('projection_sha256') == projection_sha256,
            'native_validation_partition_or_custody_invalid')


def validation_groups_informative(provenance):
    # Producer verifies groups from the frozen worker's assignment artifact.
    count = provenance.get('unique_projected_row_groups')
    require(type(count) is int and count >= 0, 'native_validation_group_receipt_invalid')
    return count >= 2


def generator_trial_eligibility(history, planned_new_trials=2):
    require(type(history) is list and type(planned_new_trials) is int and 0 <= planned_new_trials <= 2,
            'native_history_declaration_invalid')
    ids = [row.get('attempt_identity_sha256') for row in history]
    require(all(type(i) is str and len(i) == 64 and all(c in '0123456789abcdef' for c in i) for i in ids)
            and len(set(ids)) == len(ids), 'native_history_identity_invalid')
    costs = [row.get('charged_wall_seconds') for row in history]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in costs)
            and all(row.get('actual_exit_or_terminal_closed') is True
                    and row.get('failed_and_admission_wait_costs_included') is True for row in history),
            'native_history_cost_or_closure_invalid')
    require(len(history) + planned_new_trials <= 8 and sum(costs) + planned_new_trials * 600 <= 43200,
            'native_history_inclusive_budget_exhausted')
    return dict(prior_trial_count=len(history), prior_charged_wall_seconds=sum(costs),
                maximum_trials_including_prior_failures=8, maximum_cell_wall_seconds=43200)
