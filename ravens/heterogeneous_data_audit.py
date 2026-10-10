"""Audit locally generated mixed-object demonstrations before training.

Pickle files must be trusted local artifacts, never downloaded untrusted data.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def audit_dataset(data_dir, reports):
  root = Path(data_dir) / 'stack-heterogeneous-train'
  fields = ('action', 'info', 'reward', 'color', 'depth')
  names = sorted(path.name for path in (root / 'action').glob('*.pkl'))
  if not names:
    raise ValueError('No training demonstrations found')
  for field in fields:
    if sorted(path.name for path in (root / field).glob('*.pkl')) != names:
      raise ValueError('Incomplete dataset field: ' + field)
  successful = {}
  sources = []
  for path in map(Path, reports):
    raw = path.read_bytes()
    report = json.loads(raw)
    version = report['config'].get('scene_version')
    if report['config']['split'] != 'train' or version not in (None, 2):
      raise ValueError('Require scene-version-2 training reports')
    manifests = {}
    for row in report['episodes']:
      if row['success']:
        if version is None:
          # Older reports omitted the scene version; require matching saved
          # manifests rather than silently accepting an unknown distribution.
          manifest_path = path.parent / ('manifest-%d.json' % row['seed'])
          manifest_raw = manifest_path.read_bytes()
          manifest = json.loads(manifest_raw)
          if (manifest.get('schema_version') != 2 or manifest.get('split') != 'train'
              or manifest.get('seed') != row['seed']):
            raise ValueError('Incompatible legacy scene manifest')
          manifests[str(row['seed'])] = hashlib.sha256(manifest_raw).hexdigest()
        if row['seed'] in successful:
          raise ValueError('Duplicate successful seed across collection reports')
        successful[row['seed']] = row
    sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                        contact_preload=report['config'].get('contact_preload'),
                        version_evidence='config' if version else 'saved manifests',
                        manifest_sha256=manifests))
  rows, seen = [], set()
  for index, name in enumerate(names):
    number, seed_text = Path(name).stem.split('-')
    seed = int(seed_text)
    if int(number) != index or seed in seen:
      raise ValueError('Noncontiguous episode IDs or duplicate seed')
    seen.add(seed)
    if seed not in successful:
      raise ValueError('No successful training report for seed %d' % seed)
    # Only one episode's image arrays are resident at a time.
    length, actions, infos = None, None, None
    fingerprints = {}
    for field in fields:
      raw = (root / field / name).read_bytes()
      fingerprints[field] = hashlib.sha256(raw).hexdigest()
      value = pickle.loads(raw)
      if length is None:
        length = len(value)
      if len(value) != length:
        raise ValueError('Unequal observation/action lengths: ' + name)
      if field == 'action':
        actions = value
      elif field == 'info':
        infos = value
      del raw, value
    if length < 2 or actions[-1] is not None:
      raise ValueError('Missing terminal observation: ' + name)
    progress = 0
    for action, info in zip(actions[:-1], infos[:-1]):
      if not isinstance(info.get('made_progress'), bool):
        raise ValueError('Missing progress metadata: ' + name)
      progress += info['made_progress']
      for key in ('pose0', 'pose1'):
        xyz, quat = map(np.asarray, action[key])
        if (xyz.shape != (3,) or quat.shape != (4,) or
            not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)) or
            not np.isclose(np.linalg.norm(quat), 1., atol=1e-4)):
          raise ValueError('Invalid action pose: ' + name)
    if not progress:
      raise ValueError('No progress-making action: ' + name)
    rows.append(dict(file=name, seed=seed, actions=length-1,
                     progress_actions=progress, sha256=fingerprints))
  return dict(schema_version=1, dataset=str(root), episodes=len(rows),
              actions=sum(row['actions'] for row in rows),
              progress_actions=sum(row['progress_actions'] for row in rows),
              sources=sources, demonstrations=rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data_dir', required=True)
  parser.add_argument('--report', action='append', required=True)
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  output = Path(args.output)
  if output.exists():
    parser.error('Audit output exists; choose a new file')
  result = audit_dataset(args.data_dir, args.report)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(json.dumps({key: result[key] for key in
                    ('dataset', 'episodes', 'actions', 'progress_actions')}))


if __name__ == '__main__':
  main()
