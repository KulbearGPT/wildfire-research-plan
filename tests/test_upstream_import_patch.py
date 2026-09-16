"""Exercise the tracked patch against exact pinned upstream import bytes."""
from pathlib import Path
import subprocess
import tempfile
import unittest

PATCH = Path(__file__).resolve().parents[1] / 'reproductions/wsts_res18_unet_t1/patches/res18_import_scope.patch'
# ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad:src/models/__init__.py
PINNED = (b'from .BaseModel import BaseModel\n'
          b'from .ConvLSTMLightning import ConvLSTMLightning\n'
          b'from .LogisticRegression import LogisticRegression\n'
          b'from .SMPModel import SMPModel\n'
          b'from .UTAELightning import UTAELightning\n'
          b'from .SwinUnetLightning import SwinUnetLightning\n'
          b'from .SwinUnetTempLightning import SwinUnetTempLightning\n'
          b'from .UTAELightningDumb import UTAELightningDumb\n'
          b'from .TransUnetLightning import TransUnetLightning\n'
          b'from .SMPTempModel import SMPTempModel \n'
          b'from .SegFormerLightning import SegFormerLightning')


class UpstreamPatchTests(unittest.TestCase):
    def test_exact_pinned_no_newline_source_applies_and_reverses(self):
        with tempfile.TemporaryDirectory(prefix='upstream patch ') as directory:
            root = Path(directory)
            target = root / 'src/models/__init__.py'
            target.parent.mkdir(parents=True)
            target.write_bytes(PINNED)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            check = subprocess.run(['git', '-C', str(root), 'apply', '--check', str(PATCH)],
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            subprocess.run(['git', '-C', str(root), 'apply', str(PATCH)], check=True)
            self.assertEqual(target.read_bytes(), b'\n'.join(PINNED.split(b'\n')[:4]) + b'\n')
            subprocess.run(['git', '-C', str(root), 'apply', '--whitespace=nowarn', '--reverse', str(PATCH)], check=True)
            self.assertEqual(target.read_bytes(), PINNED)

    def test_compatibility_repair_is_idempotent_and_rejects_unrelated_file_drift(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('prepare_upstream',
            PATCH.parents[3] / 'scripts/research/prepare-upstream.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / module.MODEL_PATH
            dataset = root / module.DATASET_PATH
            models.parent.mkdir(parents=True)
            dataset.parent.mkdir(parents=True)
            models.write_bytes(PINNED)
            original_dataset = module.T_CO + b'class Fixture: pass\n'
            dataset.write_bytes(original_dataset)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            module.apply_known_changes(root, PATCH, PINNED, original_dataset)
            wanted = dataset.read_bytes()
            self.assertNotIn(module.T_CO, wanted)
            module.apply_known_changes(root, PATCH, PINNED, original_dataset)
            self.assertEqual(dataset.read_bytes(), wanted)
            dataset.write_bytes(wanted + b'# unrelated change\n')
            with self.assertRaisesRegex(ValueError, 'unrecognized dataset changes'):
                module.apply_known_changes(root, PATCH, PINNED, original_dataset)


if __name__ == '__main__':
    unittest.main()
