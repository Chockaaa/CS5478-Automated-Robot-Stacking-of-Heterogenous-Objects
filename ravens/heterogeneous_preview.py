"""Preview a seeded mixed-object scene; no robot actions or success claims."""
import argparse
import json
from pathlib import Path
import pybullet as p
from PIL import Image
from ravens.environments.environment import Environment
from ravens.tasks.stack_heterogeneous import SPLITS, StackHeterogeneous


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--split', choices=list(SPLITS), default='train')
  parser.add_argument('--output', default='robotiq-results/heterogeneous')
  args = parser.parse_args()
  if args.seed < 0:
    parser.error('seed must be non-negative')
  task = StackHeterogeneous(seed=args.seed, split=args.split)
  assets = Path(__file__).parent / 'environments' / 'assets'
  env = Environment(str(assets), task=task)
  try:
    env.seed(args.seed)
    env.reset()
    # Explicit settling for the preview; this is not a stability-success test.
    for _ in range(240):
      p.stepSimulation()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = '%s-%d' % (args.split, args.seed)
    Image.fromarray(env.render()).save(str(out / (stem + '.png')))
    (out / (stem + '.json')).write_text(
        json.dumps(task.manifest, indent=2), encoding='utf-8')
    print('Scene preview and manifest: %s (no actions executed)' % (out / stem))
  finally:
    env.close()


if __name__ == '__main__':
  main()
