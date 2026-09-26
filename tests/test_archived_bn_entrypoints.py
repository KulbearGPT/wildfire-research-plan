"""Standard-library checks only; the archived diagnostics never execute here."""
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AUDITS = {
    'audit_bn_activations': 'bn-activation-audit.json',
    'audit_bn_statistics': 'bn-statistics-audit.json',
}


@contextlib.contextmanager
def forbid_runtime_access():
    original_import = builtins.__import__

    def safe_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'numpy', 'h5py', 'pandas'} or name in {
            'reproductions.cross_history.data',
            'reproductions.cross_history.run_three_directions',
        }:
            raise AssertionError(f'numerical/model import attempted: {name}')
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=safe_import), \
         patch.object(Path, 'open', side_effect=AssertionError('file read/write attempted')), \
         patch.object(Path, 'exists', side_effect=AssertionError('checkpoint lookup attempted')), \
         patch.object(Path, 'mkdir', side_effect=AssertionError('directory creation attempted')):
        yield


def load_audit(name):
    path = ROOT / 'reproductions/cross_history' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchivedBatchNormEntrypointTests(unittest.TestCase):
    def test_import_and_help_do_not_touch_runtime(self):
        for name in AUDITS:
            with self.subTest(audit=name), forbid_runtime_access():
                module = load_audit(name)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stop:
                    module.main(['--help'])
                self.assertEqual(stop.exception.code, 0)
                self.assertIn('archived', output.getvalue().lower())
                self.assertIn('historical', output.getvalue().lower())
                self.assertIn('--run-root', output.getvalue())

    def test_root_is_required_without_dispatch(self):
        for name in AUDITS:
            with self.subTest(audit=name), forbid_runtime_access():
                module = load_audit(name)
                with patch.object(module, 'run') as run, \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
                    module.main([])
                self.assertEqual(stop.exception.code, 2)
                run.assert_not_called()

    def test_historical_paths_are_relative_to_explicit_root(self):
        for name, filename in AUDITS.items():
            with self.subTest(audit=name), forbid_runtime_access():
                module = load_audit(name)
                with patch.object(module, 'run') as run:
                    module.main(['--run-root', '/chosen archive/runs'])
                args = run.call_args.args[0]
                self.assertEqual(args.checkpoints, [
                    (1, Path('/chosen archive/runs/three-directions-1-bn-21787290/result/checkpoint.pt')),
                    (5, Path('/chosen archive/runs/three-directions-5-bn-21787291/result/checkpoint.pt')),
                ])
                self.assertEqual(args.output, Path('/chosen archive/runs/three-directions-analysis') / filename)

    def test_explicit_run_directories_and_output_are_forwarded(self):
        for name in AUDITS:
            with self.subTest(audit=name), forbid_runtime_access():
                module = load_audit(name)
                with patch.object(module, 'run') as run:
                    module.main([
                        '--run-root', '/chosen archive/runs', '--t1-run', 'copied t1',
                        '--t5-run', '/separate archive/copied t5', '--output', '/reports/bn audit.json',
                    ])
                args = run.call_args.args[0]
                self.assertEqual(args.checkpoints, [
                    (1, Path('/chosen archive/runs/copied t1/result/checkpoint.pt')),
                    (5, Path('/separate archive/copied t5/result/checkpoint.pt')),
                ])
                self.assertEqual(args.output, Path('/reports/bn audit.json'))

    def test_no_allocation_refuses_before_runtime_import_or_file_access(self):
        for name in AUDITS:
            with self.subTest(audit=name), forbid_runtime_access():
                module = load_audit(name)
                with patch.dict(module.os.environ, {}, clear=True), \
                     self.assertRaisesRegex(SystemExit, 'Slurm allocation required'):
                    module.main(['--run-root', '/not read'])


if __name__ == '__main__':
    unittest.main()
