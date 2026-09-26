# Code lifecycle: active_shared. Required by the mainline; optional historical branches remain archived.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
"""Checkpoint provenance checks independent of tensor/model imports."""


def require_formal_checkpoint(payload):
    """Reject smoke/incomplete checkpoints before evaluation or continuation.

    Historical formal checkpoints lack the new smoke/completed_steps fields;
    those remain compatible. New checkpoints must report their actual budget.
    """
    if payload.get('smoke', False) is not False:
        raise ValueError('smoke checkpoint cannot be used for evaluation or continuation')
    if 'completed_steps' in payload:
        completed = payload['completed_steps']
        requested = payload.get('steps')
        if (type(completed) is not int or type(requested) is not int
                or requested <= 0 or completed != requested):
            raise ValueError('checkpoint did not complete its requested training steps')


def require_evaluation_identity(payload, *, history, method, seed, block_fraction):
    """Prevent evaluation CLI metadata from relabelling the trained checkpoint.

    The archived normalized-inpaint evaluation intentionally transforms a control
    model. It is the sole method-name exception; seed and severity still match.
    Missing legacy block_fraction means the original mixed-severity setting.
    """
    for key, expected in (('history', history), ('seed', seed)):
        if type(payload.get(key)) is not int or payload[key] != expected:
            raise ValueError(f'checkpoint {key} mismatch')
    compatible_transform = method == 'normalized_inpaint' and payload.get('method') == 'control'
    if payload.get('method') != method and not compatible_transform:
        raise ValueError('checkpoint method mismatch')
    if payload.get('block_fraction') != block_fraction:
        raise ValueError('checkpoint block_fraction mismatch')
