"""Dataset integrity and split-provenance checks using tiny local fixtures."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from ravens.heterogeneous_data_audit import audit_dataset


class DatasetAuditTest(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.dataset = self.root / 'stack-heterogeneous-train'
    action = dict(pose0=((.4, -.2, .04), (0, 0, 0, 1)),
                  pose1=((.5, .18, 0), (0, 0, 0, 1)))
    for field, values in dict(action=[action, None],
        info=[dict(made_progress=True), {}], reward=[0, 1],
        color=[[], []], depth=[[], []]).items():
      (self.dataset / field).mkdir(parents=True)
      (self.dataset / field / '000000-7.pkl').write_bytes(pickle.dumps(values))
    self.report = self.root / 'results.json'
    self.report.write_text(json.dumps(dict(
        config=dict(split='train', scene_version=2, contact_preload=.02),
        episodes=[dict(seed=7, success=True)])), encoding='utf-8')

  def test_valid_dataset_records_hashes(self):
    result = audit_dataset(self.root, [self.report])
    self.assertEqual(result['episodes'], 1)
    self.assertEqual(result['progress_actions'], 1)
    self.assertEqual(len(result['demonstrations'][0]['sha256']), 5)

  def test_incomplete_fields_rejected(self):
    (self.dataset / 'depth' / '000000-7.pkl').unlink()
    with self.assertRaisesRegex(ValueError, 'Incomplete'):
      audit_dataset(self.root, [self.report])

  def test_validation_data_rejected(self):
    report = json.loads(self.report.read_text(encoding='utf-8'))
    report['config']['split'] = 'validation'
    self.report.write_text(json.dumps(report), encoding='utf-8')
    with self.assertRaisesRegex(ValueError, 'training reports'):
      audit_dataset(self.root, [self.report])

  def test_unsuccessful_demonstration_rejected(self):
    report = json.loads(self.report.read_text(encoding='utf-8'))
    report['episodes'][0]['success'] = False
    self.report.write_text(json.dumps(report), encoding='utf-8')
    with self.assertRaisesRegex(ValueError, 'No successful'):
      audit_dataset(self.root, [self.report])

  def test_missing_progress_metadata_rejected(self):
    (self.dataset / 'info' / '000000-7.pkl').write_bytes(pickle.dumps([{}, {}]))
    with self.assertRaisesRegex(ValueError, 'progress metadata'):
      audit_dataset(self.root, [self.report])

  def test_legacy_report_requires_matching_manifest(self):
    report = json.loads(self.report.read_text(encoding='utf-8'))
    del report['config']['scene_version']
    self.report.write_text(json.dumps(report), encoding='utf-8')
    with self.assertRaises(FileNotFoundError):
      audit_dataset(self.root, [self.report])
    (self.root / 'manifest-7.json').write_text(json.dumps(dict(
        schema_version=2, seed=7, split='train')), encoding='utf-8')
    result = audit_dataset(self.root, [self.report])
    self.assertEqual(result['sources'][0]['version_evidence'], 'saved manifests')


if __name__ == '__main__':
  unittest.main()
