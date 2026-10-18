"""Post-analysis of completed U-Net resolution arms against existing SAM scores."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'runs/model_comparison/unet_resolution_20261017'
RESULTS = ROOT/'results/model_comparison/unet_resolution_20261017'
SAM = ROOT/'results/phase_b/full_dev_20261012/probes'


def read(path, ids, column):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    values = {row['id']: float(row[column]) for row in rows}
    if len(values) != len(rows) or set(values) != set(ids):
        raise ValueError(f'Expected identical unique IDs: {path}')
    return np.array([values[i] for i in ids])


def interval(values):
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'ci95': [float(lo), float(hi)]}


def main():
    status = json.loads((RESULTS/'status.json').read_text())
    if status != {'state': 'complete', 'completed_runs': 6, 'test_evaluated': False}:
        raise ValueError('Six completed dev-only runs required')
    plan = json.loads((RUNS/'plan.json').read_text())
    ids = plan['ids']['dev']
    sam = np.mean([read(SAM/f'seed{s}/probe_dev_per_image.csv', ids, 'selfboxpoint_dice')
                   for s in plan['seeds']], axis=0)
    vectors = {'sam_auto': sam}
    for size in plan['sizes']:
        for filename, label in [('endpoint_dev_per_image.csv', 'endpoint'),
                                ('dev_per_image.csv', 'selected')]:
            vectors[f'unet_{size}_{label}'] = np.mean([
                read(RUNS/f'size{size}/seed{s}/{filename}', ids, 'dice')
                for s in plan['seeds']], axis=0)
    comparisons = {}
    for name in ['unet_128_endpoint', 'unet_128_selected',
                 'unet_256_endpoint', 'unet_256_selected']:
        comparisons[f'sam_auto minus {name}'] = interval(sam-vectors[name])
    summaries = json.loads((RESULTS/'summary.json').read_text())['runs']
    latency = {str(size): {
        'median_ms_across_seed_medians': float(np.median([
            summaries[f'{size}/seed{s}']['latency']['median_ms'] for s in plan['seeds']])),
        'seed_medians_ms': [summaries[f'{size}/seed{s}']['latency']['median_ms']
                            for s in plan['seeds']]}
        for size in plan['sizes']}
    output = {'test_evaluated': False, 'n_dev': len(ids), 'seeds': plan['seeds'],
              'bootstrap': '10000 paired image resamples after seed averaging; RNG seed 0',
              'comparisons': comparisons, 'unet_latency': latency,
              'warning': 'Exploratory development comparison; intervals do not establish equivalence.'}
    (RESULTS/'comparison_with_sam.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
