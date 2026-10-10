"""Protect existing checkpoints before any model construction or training."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from ravens import heterogeneous_train


class CheckpointProtectionTest(unittest.TestCase):
  def test_existing_checkpoints_are_not_overwritten(self):
    with tempfile.TemporaryDirectory() as root:
      folder = Path(root) / 'checkpoints' / 'mixed-tower-v2'
      folder.mkdir(parents=True)
      checkpoint = folder / 'attention-ckpt-1000.h5'
      checkpoint.write_bytes(b'checkpoint fixture')
      for resume, message in ((0, 'Checkpoints already exist'),
                              (400, 'overwrite a later checkpoint')):
        stderr = io.StringIO()
        argv = ['train', '--train_dir=' + root, '--resume=' + str(resume),
                '--steps=1000', '--save_every=200']
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stderr(stderr):
          with self.assertRaises(SystemExit) as caught:
            heterogeneous_train.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, stderr.getvalue())
        self.assertEqual(checkpoint.read_bytes(), b'checkpoint fixture')


if __name__ == '__main__':
  unittest.main()
