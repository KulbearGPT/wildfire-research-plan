"""Execute the real loader body with stdlib model boundaries, without torch."""
import ast
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'reproductions/wsts_fast_track'


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class StateHolder:
    def load_state_dict(self, state, *, strict):
        if not strict:
            raise AssertionError('state restoration must be strict')
        self.state = dict(state)


class Base(StateHolder):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Adapter:
    def __init__(self, base, *, adapter_scope):
        self.base_model = base
        self.adapter_scope = adapter_scope
        self.adapter = StateHolder()

    def to(self, device):
        self.device = device
        return self


class CraPortabilityTests(unittest.TestCase):
    def test_embedded_base_and_adapter_load_without_historical_b3(self):
        namespace = dict(Path=Path, sys=sys, importlib=importlib,
                         CounterfactualReliabilityAdapter=Adapter,
                         load_checkpoint_model=lambda *args, **kwargs: self.fail('external checkpoint read'))
        load_function(ROOT / 'evaluate_missingness.py', 'checkpoint_init_args', namespace)
        loader = load_function(ROOT / 'evaluate_counterfactual_reliability_adapter.py',
                               'load_cra_model', namespace)
        upstream = types.ModuleType('models.SMPModel')
        upstream.SMPModel = Base
        original_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'models.SMPModel': upstream}):
                missing = str(Path(directory) / 'missing-original-b3.ckpt')
                for candidate, scope in [('D7-CRA', 'all'), ('D8-FFCA', 'block')]:
                    payload = dict(candidate_id=candidate, base_b3_checkpoint=missing,
                                   hyper_parameters={'encoder_weights': 'imagenet', 'n_channels': 40},
                                   base_state_dict={'weight': 7}, adapter_state_dict={'weight': 11})
                    try:
                        model = loader(payload, upstream_root=Path(directory), device='cpu')
                    except FileNotFoundError:
                        self.fail('loader still requires missing historical B3 checkpoint')
                    self.assertEqual(model.base_model.state, {'weight': 7})
                    self.assertEqual(model.adapter.state, {'weight': 11})
                    self.assertEqual(model.base_model.kwargs['encoder_weights'], None)
                    self.assertEqual(model.adapter_scope, scope)
                    self.assertEqual(payload['base_b3_checkpoint'], missing)
                    self.assertFalse(Path(missing).exists())
        finally:
            sys.path[:] = original_path


if __name__ == '__main__':
    unittest.main()
