"""Shell boundary checks; no scheduler or tensor dependencies."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResearchLauncherTests(unittest.TestCase):
    def test_job_refuses_setup_outside_slurm_before_sourcing_site(self):
        script = ROOT / 'scripts/research/job.sh'
        self.assertTrue(script.is_file(), 'generic allocation runner is missing')
        with tempfile.TemporaryDirectory(prefix='research site ') as directory:
            site = Path(directory) / 'site.env'
            marker = Path(directory) / 'sourced'
            site.write_text(f'touch "{marker}"\n')
            env = dict(os.environ)
            env.pop('SLURM_JOB_ID', None)
            result = subprocess.run(['bash', str(script), str(ROOT), str(site), 'setup'],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Slurm allocation required', result.stderr)
            self.assertFalse(marker.exists())

    def test_submit_validates_arguments(self):
        script = ROOT / 'scripts/research/submit.sh'
        self.assertTrue(script.is_file(), 'generic submission script is missing')
        result = subprocess.run(['bash', str(script), 'invalid', 'setup'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('usage:', result.stderr)

    def test_submit_archives_commit_and_preserves_paths_and_arguments(self):
        import shutil
        import shlex
        with tempfile.TemporaryDirectory(prefix='research launch ') as directory:
            base = Path(directory)
            repo = base / 'repository with spaces'
            repo.mkdir()
            shutil.copytree(ROOT / 'scripts/research', repo / 'scripts/research')
            def git(*args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()
            git('init')
            git('add', '.')
            git('-c', 'user.name=Shell Test', '-c', 'user.email=test@example.invalid',
                'commit', '-m', 'test snapshot')
            commit = git('rev-parse', 'HEAD')
            artifact_root = base / 'artifacts with spaces'
            site = base / 'site with spaces.env'
            site.write_text('WILDFIRE_REPO=' + shlex.quote(str(repo)) + '\n' +
                            'WILDFIRE_ROOT=' + shlex.quote(str(artifact_root)) + '\n' +
                            'WILDFIRE_CPU_SBATCH=("--comment=two words" "--mem=1G")\n')
            fake_bin = base / 'bin'
            fake_bin.mkdir()
            scheduler_args = base / 'scheduler args'
            fake = fake_bin / 'sbatch'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$TEST_SCHEDULER_ARGS"\nprintf "12345\\n"\n')
            fake.chmod(0o755)
            env = dict(os.environ, WILDFIRE_SITE_ENV=str(site),
                       TEST_SCHEDULER_ARGS=str(scheduler_args),
                       PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT / 'scripts/research/submit.sh'),
                                     'cpu', 'audit', 'python', '-c', 'print("two words")'],
                                    env=env, capture_output=True, text=True, check=True)
            run, = (artifact_root / 'jobs').iterdir()
            self.assertEqual((run / 'source-commit.txt').read_text().strip(), commit)
            self.assertEqual((run / 'source/scripts/research/job.sh').read_bytes(),
                             (repo / 'scripts/research/job.sh').read_bytes())
            args = scheduler_args.read_bytes().decode().split('\0')[:-1]
            self.assertIn('--comment=two words', args)
            self.assertEqual(args[-4:], ['audit', 'python', '-c', 'print("two words")'])
            self.assertIn(str(run / 'source'), args)
            self.assertIn(str(run / 'site-at-submission.env'), args)
            self.assertIn('Submitted 12345', result.stdout)

    def test_modules_and_activation_cannot_restore_vendor_pip_or_python_paths(self):
        import shlex
        for action in ('setup', 'runtime'):
            with self.subTest(action=action), tempfile.TemporaryDirectory(prefix='module reset ') as directory:
                base = Path(directory)
                snapshot = base / 'source'
                snapshot.mkdir()
                (base / 'source-commit.txt').write_text('a' * 40 + '\n')
                fake_bin = base / 'bin'
                fake_bin.mkdir()
                observed = base / 'observed.env'
                fake_python = fake_bin / 'python'
                fake_python.write_text('#!/usr/bin/env bash\nenv > "$TEST_OBSERVED"\nexit 77\n')
                fake_python.chmod(0o755)
                root = base / 'artifacts'
                injection = ('export PIP_NO_INDEX=1 PIP_FIND_LINKS=/vendor/wheels '
                             'PIP_EXTRA_INDEX_URL=https://vendor.invalid/simple '
                             'PIP_INDEX_URL=https://vendor.invalid/index '
                             'PIP_CONFIG_FILE=/vendor/pip.conf '
                             'PYTHONPATH=/old/worktree PYTHONHOME=/vendor/python\n')
                if action == 'runtime':
                    activate = root / 'envs/train/bin/activate'
                    activate.parent.mkdir(parents=True)
                    activate.write_text(injection)
                site = base / 'site.env'
                site.write_text('WILDFIRE_REPO=/original/checkout\n' +
                    'WILDFIRE_ROOT=' + shlex.quote(str(root)) + '\n' +
                    'WILDFIRE_UPSTREAM=' + shlex.quote(str(root / 'upstream')) + '\n' +
                    'WILDFIRE_DATA=/unused/data\nWILDFIRE_STATS=/unused/stats\n' +
                    'WILDFIRE_TRAIN_MODULES=fake\n' +
                    'WILDFIRE_TRAIN_PYTHON=' + shlex.quote(str(fake_python)) + '\n' +
                    'scontrol() { :; }\nmodule() { ' + injection + '}\n')
                env = dict(os.environ, SLURM_JOB_ID='boundary-test', TEST_OBSERVED=str(observed),
                           PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
                result = subprocess.run(['bash', str(ROOT / 'scripts/research/job.sh'),
                    str(snapshot), str(site), 'setup' if action == 'setup' else 'true'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 77, result.stderr)
                values = dict(line.split('=', 1) for line in observed.read_text().splitlines() if '=' in line)
                self.assertEqual(values['PIP_INDEX_URL'], 'https://pypi.org/simple')
                self.assertEqual(values['PIP_NO_INDEX'], '0')
                self.assertEqual(values['PIP_CONFIG_FILE'], '/dev/null')
                self.assertNotIn('PIP_FIND_LINKS', values)
                self.assertNotIn('PIP_EXTRA_INDEX_URL', values)
                self.assertNotIn('PYTHONHOME', values)
                self.assertEqual(values['PYTHONPATH'],
                                 f'{snapshot}/src:{snapshot}:{root}/upstream/src')
                self.assertEqual(values['WILDFIRE_REPO'], str(snapshot))
                self.assertEqual(values.get('WILDFIRE_SOURCE_COMMIT'), 'a' * 40)
                self.assertEqual(values['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'], '1')

    def test_setup_finish_requires_existing_training_environment(self):
        import shlex
        with tempfile.TemporaryDirectory(prefix='setup finish ') as directory:
            root = Path(directory)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            site = root / 'site.env'
            site.write_text('WILDFIRE_REPO=/unused\nWILDFIRE_ROOT=' + shlex.quote(str(root)) +
                            '\nWILDFIRE_UPSTREAM=/unused/upstream\nWILDFIRE_DATA=/unused/data\n'
                            'WILDFIRE_STATS=/unused/stats\nscontrol() { :; }\n')
            result = subprocess.run(['bash', str(ROOT / 'scripts/research/job.sh'),
                                     str(snapshot), str(site), 'setup-finish'],
                                    env=dict(os.environ, SLURM_JOB_ID='test-boundary'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires the existing training environment', result.stderr)
            self.assertFalse((root / 'envs').exists())


if __name__ == '__main__':
    unittest.main()
