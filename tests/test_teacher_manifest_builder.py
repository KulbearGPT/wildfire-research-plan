"""Manifest metadata/hash checks without importing torch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from reproductions import build_teacher_manifest as builder
except ImportError:
    builder = None


class TeacherManifestTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(builder, 'teacher manifest builder is missing')
        self.payload = dict(history=1, seed=0, method='control', architecture='res18_unet',
                            steps=3000, physical_batch=64, effective_batch=64,
                            block_fraction=None, initial_checkpoint='/new/base.ckpt')
        self.summary = dict(history=1, seed=0, method='control', architecture='res18_unet',
                            year=2021, block_fraction=None,
                            results={key: {'sample_count': 3181} for key in ('M00','M01','M06','M07')})

    def test_rejects_wrong_source_contract_and_summary(self):
        for key, value in [('history',5), ('seed',1), ('method','cosine_erm'),
                           ('steps',1), ('physical_batch',16), ('effective_batch',32),
                           ('block_fraction',.25), ('architecture','res18_utae'),
                           ('initial_checkpoint','')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                builder.validate_source(dict(self.payload, **{key: value}), self.summary,
                                        history=1, seed=0, method='control', block_fraction=None)
        with self.assertRaises(ValueError):
            builder.validate_source(self.payload, dict(self.summary, year=2022),
                                    history=1, seed=0, method='control', block_fraction=None)

    def test_relative_artifacts_fresh_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/t1-s0-control'
            run.mkdir(parents=True)
            checkpoint = run / 'checkpoint.pt'
            checkpoint.write_bytes(b'newly generated checkpoint')
            (run / 'summary.json').write_text(json.dumps(self.summary))
            output = root / 'manifests/teachers.json'
            record = builder.source_record(self.payload, self.summary, checkpoint, output,
                                            history=1, seed=0, method='control', block_fraction=None)
            self.assertEqual(record['checkpoint'], '../runs/t1-s0-control/checkpoint.pt')
            self.assertEqual(record['summary'], '../runs/t1-s0-control/summary.json')
            self.assertEqual(record['sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(record['bytes'], checkpoint.stat().st_size)
            self.assertEqual(record['metadata']['initial_checkpoint'], '/new/base.ckpt')

    def test_no_checkpoint_load_outside_slurm(self):
        env = dict(os.environ)
        env.pop('SLURM_JOB_ID', None)
        result = subprocess.run([sys.executable, '-S', '-m', 'reproductions.build_teacher_manifest',
            '--runs-root', '/unused', '--output', '/unused/output.json', '--format', 'reliability'],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Slurm allocation required', result.stderr)
        self.assertNotIn("No module named 'torch'", result.stderr)


if __name__ == '__main__':
    unittest.main()
