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
