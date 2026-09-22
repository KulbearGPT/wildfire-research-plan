"""Control-plane checks only: no models, tensors, datasets or scheduler calls."""
import contextlib
import io
import unittest
from unittest.mock import patch

from reproductions.cross_history import mainline


class MainlineCommandTests(unittest.TestCase):
    def args(self, *extra):
        return ['--history', '1', '--method', 'control', '--output', 'a path/run', *extra]

    def test_all_five_mainline_roles(self):
        for history in (1, 5):
            for method, fraction in [('control', None), ('cosine_erm', None),
                                     ('block_specialist', None),
                                     ('block_specialist', '0.25'),
                                     ('block_specialist', '0.5')]:
                args = ['--history', str(history), '--method', method, '--output', 'run']
                if fraction:
                    args += ['--block-fraction', fraction]
                _, forwarded = mainline.command(args)
                options = dict(zip(forwarded[::2], forwarded[1::2]))
                self.assertEqual(options['--steps'], '3000')
                self.assertEqual(options['--batch-size'], '64')
                self.assertEqual(options['--method'], method)
                self.assertEqual(options.get('--block-fraction'), fraction)
                self.assertNotIn('--architecture', options)

    def test_invalid_or_archived_requests_do_not_dispatch(self):
        for extra in [('--method', 'impact_consistency'), ('--architecture', 'swin_unet'),
                      ('--batch-size', '16'), ('--steps', '100'), ('--seed', '3'),
                      ('--workers', '-1'), ('--block-fraction', '0.25'),
                      ('--year', '2022'), ('--bootstrap',),
                      ('--smoke', '--evaluate-only', 'checkpoint.pt')]:
            with self.subTest(extra=extra), patch.object(mainline.runpy, 'run_module') as run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    mainline.main(self.args(*extra))
                run.assert_not_called()

    def test_print_and_missing_allocation_do_not_load_trainer(self):
        with patch.dict(mainline.os.environ, {}, clear=True), patch.object(mainline.runpy, 'run_module') as run:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mainline.main(self.args('--print-command'))
            self.assertIn("'a path/run'", output.getvalue())
            with self.assertRaisesRegex(SystemExit, 'Slurm allocation required'):
                mainline.main(self.args())
            run.assert_not_called()

    def test_evaluation_dispatch_and_argv_restoration(self):
        previous = mainline.sys.argv
        observed = []
        def capture(*args, **kwargs):
            observed.extend(mainline.sys.argv)
            raise RuntimeError('mock trainer failure')
        with patch.dict(mainline.os.environ, {'SLURM_JOB_ID': 'mock-only'}), \
             patch.object(mainline.runpy, 'run_module', side_effect=capture) as run:
            with self.assertRaisesRegex(RuntimeError, 'mock trainer failure'):
                mainline.main(self.args('--year', '2023', '--evaluate-only', 'a path/checkpoint.pt'))
            run.assert_called_once_with('reproductions.cross_history.run', run_name='__main__')
        self.assertIn('a path/checkpoint.pt', observed)
        self.assertEqual(observed[observed.index('--year') + 1], '2023')
        self.assertIs(mainline.sys.argv, previous)


if __name__ == '__main__':
    unittest.main()
