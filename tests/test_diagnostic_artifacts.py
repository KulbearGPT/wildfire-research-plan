"""Stdlib relocation checks; these do not import model/tensor code."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from reproductions.wsts_fast_track.diagnostic_artifacts import resolve_artifact, provenance_matches, verified_reliability_path


class DiagnosticArtifactsTests(unittest.TestCase):
    def test_relocated_relative_file_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'moved.pt'
            target.write_bytes(b'original')
            manifest = root / 'map.json'
            manifest.write_text(json.dumps(dict(schema_version=1, artifacts={
                '/unavailable/old.pt': dict(path='moved.pt', bytes=8,
                    sha256=hashlib.sha256(b'original').hexdigest())})))
            self.assertEqual(resolve_artifact('/unavailable/old.pt', artifact_map=manifest), target)
            self.assertTrue(provenance_matches('/unavailable/old.pt', target, artifact_map=manifest))
            self.assertFalse(provenance_matches('/unavailable/old.pt', target))
            with self.assertRaisesRegex(ValueError, 'lacks recorded provenance'):
                resolve_artifact('/unavailable/other.pt', artifact_map=manifest)
            target.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'changed'):
                resolve_artifact('/unavailable/old.pt', artifact_map=manifest)

    def test_original_relative_record_resolves_beside_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'weight.pt').write_bytes(b'checkpoint')
            self.assertEqual(resolve_artifact('weight.pt', relative_to=root), root / 'weight.pt')

    def test_byte_size_is_checked_even_when_digest_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root / 'w'; target.write_bytes(b'original')
            manifest = root / 'map.json'
            manifest.write_text(json.dumps(dict(schema_version=1, artifacts={
                '/old': dict(path='w', bytes=9, sha256=hashlib.sha256(b'original').hexdigest())})))
            with self.assertRaisesRegex(ValueError, 'changed'):
                resolve_artifact('/old', artifact_map=manifest)

    def test_reliability_digest_verified_before_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'map.npz'
            path.write_bytes(b'map data')
            record = dict(output='map.npz', sha256=hashlib.sha256(b'map data').hexdigest())
            self.assertEqual(verified_reliability_path(root, record), path)
            path.write_bytes(b'bad data')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verified_reliability_path(root, record)
            with self.assertRaisesRegex(ValueError, 'valid SHA256'):
                verified_reliability_path(root, dict(output='map.npz'))

    def test_reliability_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / 'outside.npz'
            outside.write_bytes(b'map')
            nested = root / 'nested'
            nested.mkdir()
            with self.assertRaisesRegex(ValueError, 'within its root'):
                verified_reliability_path(nested, dict(output='../outside.npz',
                    sha256=hashlib.sha256(b'map').hexdigest()))
