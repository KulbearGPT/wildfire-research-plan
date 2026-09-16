"""Standard-library checks of preparation entrypoint boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/research/prepare-data.sh'


class PreparationTests(unittest.TestCase):
    def test_download_refuses_login_before_creating_files(self):
        self.assertTrue(SCRIPT.is_file(), 'portable data preparation entrypoint missing')
        with tempfile.TemporaryDirectory(prefix='data root ') as directory:
            env = dict(os.environ, WILDFIRE_ROOT=directory)
            env.pop('SLURM_JOB_ID', None)
            result = subprocess.run(['bash', str(SCRIPT), 'download', 'original'],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Slurm allocation required', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unknown_stage_fails_before_data_access(self):
        self.assertTrue(SCRIPT.is_file(), 'portable data preparation entrypoint missing')
        result = subprocess.run(['bash', str(SCRIPT), 'unknown'],
                                env=dict(os.environ, SLURM_JOB_ID='test-boundary'),
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('usage:', result.stderr)

    def test_wrong_environment_fails_before_preparation_output(self):
        self.assertTrue(SCRIPT.is_file())
        with tempfile.TemporaryDirectory(prefix='data root ') as directory:
            env = dict(os.environ, SLURM_JOB_ID='test-boundary', WILDFIRE_ROOT=directory,
                       WILDFIRE_REPO=str(ROOT), WILDFIRE_UPSTREAM='/unused',
                       WILDFIRE_DATA=directory + '/data', WILDFIRE_STATS=directory + '/stats.npz',
                       VIRTUAL_ENV=directory + '/envs/train')
            result = subprocess.run(['bash', str(SCRIPT), 'repair'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires the audit environment', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_rechecks_existing_archive_without_network(self):
        self.assertTrue(SCRIPT.is_file())
        with tempfile.TemporaryDirectory(prefix='data root ') as directory:
            root = Path(directory)
            (root / 'downloads').mkdir()
            archive = root / 'downloads/WildfireSpreadTS.zip'
            archive.write_bytes(b'invalid tiny archive')
            env = dict(os.environ, SLURM_JOB_ID='test-boundary', WILDFIRE_ROOT=directory,
                       WILDFIRE_REPO=str(ROOT), WILDFIRE_UPSTREAM='/unused',
                       WILDFIRE_DATA=directory + '/data', WILDFIRE_STATS=directory + '/stats.npz',
                       VIRTUAL_ENV=directory + '/envs/train')
            result = subprocess.run(['bash', str(SCRIPT), 'download', 'original'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FAILED', result.stdout)
            self.assertEqual(archive.read_bytes(), b'invalid tiny archive')
            self.assertFalse(list((root / 'preparation').glob('*/COMPLETED')))


if __name__ == '__main__':
    unittest.main()
