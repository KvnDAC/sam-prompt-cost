"""Cross-model scores must pair the same images regardless of CSV row order."""
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completed',ROOT/'scripts/23_compare_completed_models.py')
completed=importlib.util.module_from_spec(spec);spec.loader.exec_module(completed)


class CompletedComparisonTests(unittest.TestCase):
    def test_reorders_scores_by_id_and_supports_named_metrics(self):
        path=Mock(spec=Path)
        path.open.return_value=io.StringIO('id,probe_dice\nb,0.7\na,0.2\n')
        np.testing.assert_array_equal(completed.read_scores(path,['a','b'],'probe_dice'),[.2,.7])

    def test_missing_extra_and_duplicate_scores_rejected(self):
        for content in ['id,dice\na,0.2\n','id,dice\na,0.2\nb,0.7\nc,0.1\n',
                        'id,dice\na,0.2\na,0.7\n']:
            path=Mock(spec=Path);path.open.return_value=io.StringIO(content)
            with self.assertRaises(ValueError):completed.read_scores(path,['a','b'])


if __name__=='__main__':unittest.main()
