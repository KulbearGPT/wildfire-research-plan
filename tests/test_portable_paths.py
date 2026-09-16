"""Portable artifact selection tests; standard library only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from reproductions import paths
except ImportError:
    paths = None


class PortablePathsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(paths, 'central portable path configuration is missing')

    def test_explicit_configuration_reports_missing_variables(self):
        with self.assertRaisesRegex(ValueError, 'WILDFIRE_UPSTREAM'):
            paths.load_paths(require_explicit=True, environ={'WILDFIRE_ROOT': '/new'})

    def test_selected_checkpoint_is_required_and_never_falls_back(self):
        env = {f'WILDFIRE_{key}': f'/new/{key.lower()}' for key in ('ROOT', 'UPSTREAM', 'DATA', 'STATS')}
        with self.assertRaisesRegex(ValueError, 'WILDFIRE_B5_CHECKPOINT'):
            paths.load_paths(require_explicit=True, required=('b5_checkpoint',), environ=env)
        env['WILDFIRE_B5_CHECKPOINT'] = '/new/weights/b5.ckpt'
        config = paths.load_paths(require_explicit=True, required=('b5_checkpoint',), environ=env)
        self.assertEqual(config.initial_checkpoint(5), Path('/new/weights/b5.ckpt'))
        self.assertEqual(config.upstream, Path('/new/upstream'))

    def test_all_explicit_artifacts_survive_a_fresh_process(self):
        import os
        import subprocess
        import sys
        env = dict(os.environ)
        fields = ('root', 'upstream', 'data', 'stats', 'b3_checkpoint',
                  'b5_checkpoint', 'teacher_manifest')
        env.update({'WILDFIRE_' + key.upper(): '/relocated/' + key for key in fields})
        code = ("from reproductions.paths import load_paths; "
                "p=load_paths(require_explicit=True, required=('b3_checkpoint', 'b5_checkpoint', 'teacher_manifest')); "
                "print(p)")
        result = subprocess.run([sys.executable, '-S', '-c', code], env=env,
                                check=True, text=True, capture_output=True)
        for key in fields:
            self.assertIn('/relocated/' + key, result.stdout)
        self.assertNotIn('/project/', result.stdout)

    def test_teacher_size_is_checked_independently_of_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'teacher.pt'
            checkpoint.write_bytes(b'valid')
            record = dict(checkpoint=str(checkpoint), bytes=100,
                          sha256=hashlib.sha256(b'valid').hexdigest())
            with self.assertRaisesRegex(ValueError, 'teacher checkpoint changed'):
                paths.validate_teacher_artifact(record)

    def test_relative_environment_paths_resolve_from_working_directory(self):
        config = paths.load_paths(environ={'WILDFIRE_ROOT': 'artifacts'})
        self.assertEqual(config.root, Path('artifacts').resolve())
        self.assertEqual(config.upstream, config.root / 'cache/WildfireSpreadTS-res18-runtime')

    def test_manifest_relocation_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'teacher.pt'
            checkpoint.write_bytes(b'teacher fixture')
            record = dict(checkpoint='teacher.pt', summary='summary.json', bytes=15,
                          sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), method='control')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'sources': {'1': {'0': {'clean': record}}}}))
            resolved = paths.resolve_teacher_records(manifest, 1, 0)['clean']
            self.assertEqual(resolved['checkpoint'], str(checkpoint))
            self.assertEqual(resolved['summary'], str(root / 'summary.json'))
            self.assertEqual(resolved['sha256'], record['sha256'])
            self.assertEqual(paths.validate_teacher_artifact(resolved), checkpoint)
            checkpoint.write_bytes(b'teacher altered')
            with self.assertRaisesRegex(ValueError, 'teacher checkpoint changed'):
                paths.validate_teacher_artifact(resolved)

    def test_legacy_absolute_manifest_path_remains_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'sources': {'5': {'2': {'fire': {'checkpoint': '/legacy/teacher.pt'}}}}}))
            self.assertEqual(paths.resolve_teacher_records(manifest, 5, 2)['fire']['checkpoint'], '/legacy/teacher.pt')


if __name__ == '__main__':
    unittest.main()
