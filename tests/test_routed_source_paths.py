"""Manifest relocation must not rewrite checkpoint provenance identity."""
import json
from pathlib import Path
import tempfile
import unittest

from reproductions.three_directions.sources import resolve_sources


class RoutedSourcePathsTests(unittest.TestCase):
    def test_relative_paths_and_nullable_summary_preserve_identity(self):
        with tempfile.TemporaryDirectory(prefix='relocated sources ') as directory:
            root = Path(directory)
            manifest = root / 'sources.json'
            identity = '/old/site/initial.ckpt'
            manifest.write_text(json.dumps({'t5-s0': {
                'erm': {'checkpoint': 'weights/erm.pt', 'summary': None,
                        'metadata': {'initial_checkpoint': identity}},
                'x22': {'checkpoint': '/absolute/x22.pt', 'summary': 'scores.json'},
            }}))
            records = resolve_sources(manifest, 5, 0)
            self.assertEqual(records['erm']['checkpoint'], str(root / 'weights/erm.pt'))
            self.assertIsNone(records['erm']['summary'])
            self.assertEqual(records['erm']['metadata']['initial_checkpoint'], identity)
            self.assertEqual(records['x22']['checkpoint'], '/absolute/x22.pt')
            self.assertEqual(records['x22']['summary'], str(root / 'scores.json'))
