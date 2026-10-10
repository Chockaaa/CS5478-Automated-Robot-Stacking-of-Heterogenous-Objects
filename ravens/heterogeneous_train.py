"""Transporter training with cached RGB-D maps and resumable weights."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from ravens.dataset import Dataset
from ravens.agents.transporter_heterogeneous import HeterogeneousTransporterAgent
from ravens.utils import utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data_dir', default='robotiq-dataset/mixed-v2')
  parser.add_argument('--train_dir', default='robotiq-training')
  parser.add_argument('--name', default='mixed-tower-v2')
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--save_every', type=int, default=100)
  parser.add_argument('--resume', type=int, default=0)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--threads', type=int, default=4)
  parser.add_argument('--compile_models', action='store_true',
                      help='Trace the unchanged Keras forward passes to reduce eager overhead')
  parser.add_argument('--demonstration_report', help='Optional collection results for older demos without progress metadata')
  args = parser.parse_args()
  if (min(args.steps, args.save_every, args.threads) < 1 or
      not 0 <= args.resume < args.steps or args.seed < 0):
    parser.error('Require positive counts, nonnegative seed, and 0 <= resume < steps')
  checkpoint_dir = Path(args.train_dir) / 'checkpoints' / args.name
  if not args.resume and any(checkpoint_dir.glob('*.h5')):
    parser.error('Checkpoints already exist: choose a new --name or use --resume')
  planned = [step for step in range(args.resume + 1, args.steps + 1)
             if step % args.save_every == 0 or step == args.steps]
  if any((checkpoint_dir / ('%s-ckpt-%d.h5' % (module, step))).exists()
         for step in planned for module in ('attention', 'transport')):
    parser.error('This run would overwrite a later checkpoint; resume from the latest saved step')
  tf.config.threading.set_intra_op_parallelism_threads(args.threads)
  tf.config.threading.set_inter_op_parallelism_threads(2)
  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  random.seed(args.seed)
  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  agent = HeterogeneousTransporterAgent(args.name, 'stack-heterogeneous', args.train_dir)
  if args.resume:
    agent.load(args.resume)
  if args.compile_models:
    for module in (agent.attention, agent.transport):
      module.model.call = tf.function(module.model.call, reduce_retracing=True)
  dataset = Dataset(str(Path(args.data_dir) / 'stack-heterogeneous-train'))
  if not dataset.n_episodes:
    raise ValueError('Collect mixed-object demonstrations before training')
  samples, seeds = [], []
  reports = {}
  if args.demonstration_report:
    report = json.loads(Path(args.demonstration_report).read_text(encoding='utf-8'))
    if report['config']['split'] != 'train':
      raise ValueError('Demonstration report must describe the training split')
    reports = {row['seed']: row for row in report['episodes'] if row['success']}
  skipped = 0
  for index in range(dataset.n_episodes):
    episode, seed = dataset.load(index)
    seeds.append(seed)
    previous_count = 0
    for action_index, (obs, act, _, info) in enumerate(episode[:-1]):
      made_progress = info.get('made_progress', True)
      if reports:
        details = reports[seed]['actions'][action_index]
        made_progress = details['stable_objects'] > previous_count
        previous_count = details['stable_objects']
      if not made_progress:
        skipped += 1
        continue
      img = agent.get_image(obs)
      p0 = utils.xyz_to_pix(act['pose0'][0], agent.bounds, agent.pix_size)
      p1 = utils.xyz_to_pix(act['pose1'][0], agent.bounds, agent.pix_size)
      for pixel in (p0, p1):
        if not all(0 <= coord < size for coord, size in zip(pixel, agent.in_shape[:2])):
          raise ValueError('Demonstration seed %d has an action outside the image bounds' % seed)
      samples.append((img, p0, p1))
  if not samples:
    raise ValueError('No usable demonstration actions')
  folder = Path(args.train_dir) / args.name
  folder.mkdir(parents=True, exist_ok=True)
  config = dict(vars(args), dataset_seeds=seeds, samples=len(samples), skipped_nonprogress_actions=skipped,
                tensorflow_version=tf.__version__, numpy_version=np.__version__,
                devices=[d.name for d in tf.config.list_physical_devices()],
                augmentation='RGB channel permutation and gain 0.8-1.2; no spatial goal shift',
                architecture='Original Attention/Transport ResNets; 192x112, crop32, 1 rotation; RGB-D geometry yaw')
  (folder / ('config-%d.json' % args.resume)).write_text(json.dumps(config, indent=2), encoding='utf-8')
  started = time.perf_counter()
  history = []
  print('Training on %d episodes / %d actions' % (len(seeds), len(samples)), flush=True)
  for step in range(args.resume + 1, args.steps + 1):
    img, p0, p1 = samples[np.random.randint(len(samples))]
    img = img.copy()
    img[:, :, :3] = np.clip(img[:, :, np.random.permutation(3)] *
                            np.random.uniform(.8, 1.2), 0, 255)
    loss0 = agent.attention.train(img.copy(), p0, 0.)
    loss1 = agent.transport.train(img.copy(), p0, p1, 0.)
    agent.total_steps = step
    row = dict(step=step, attention_loss=float(loss0), transport_loss=float(loss1),
               seconds=time.perf_counter() - started)
    history.append(row)
    if step % 10 == 0 or step == args.resume + 1:
      print(json.dumps(row), flush=True)
    if step % args.save_every == 0 or step == args.steps:
      agent.save()
      (folder / ('history-%d.json' % args.resume)).write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
  main()
