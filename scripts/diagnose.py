#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path


def gate(reason):
    """Group by the gate that fired, keeping the numbers out of the key."""
    return (reason or 'unstated').split(':')[0].strip()[:58]


def scan(runs):
    counts = {name: Counter() for name in
              ('run_state', 'selected', 'proposal', 'proposal_gate', 'candidate',
               'submap', 'submap_gate', 'dropped')}
    example, loops = {}, Counter()
    for run in runs:
        meta = json.loads((run / 'run.json').read_text())
        counts['run_state'][f"{meta['state']}: {gate(meta.get('error'))}"
                            if meta['state'] == 'failed' else meta['state']] += 1
        counts['selected'][str(meta.get('selected'))] += 1
        counts['dropped']['optional tasks dropped'] += meta.get('dropped_optional_tasks', 0)
        counts['dropped']['optional tasks submitted'] += meta.get('submitted_optional_tasks', 0)
        for line in (run / 'events.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event['kind'] == 'candidate_rejected':
                counts['candidate'][gate(event.get('reason'))] += 1
            if event['kind'] != 'vision':
                continue
            result = event['result']
            if result.get('kind') == 'error':
                counts['run_state']['vision worker died'] += 1
                continue
            payload, kind = result.get('payload', {}), result.get('kind')
            status = payload.get('status', 'unknown')
            key = 'proposal' if kind == 'assist' else 'submap'
            counts[key][status] += 1
            if status in ('rejected', 'deferred'):
                counts[key + '_gate'][gate(payload.get('reason'))] += 1
                example.setdefault(gate(payload.get('reason')), payload.get('reason', ''))
            if status == 'mapped':
                loops['accepted'] += payload.get('accepted_loops', 0)
    return counts, example, loops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True,
                   help='Run directories, or matrix directories containing them')
    p.add_argument('--top', type=int, default=8)
    a = p.parse_args()
    runs = []
    for path in a.runs:
        runs += [path] if (path / 'run.json').exists() else sorted(
            child.parent for child in path.glob('*/run.json'))
    if not runs:
        raise SystemExit('No run directories found')
    counts, example, loops = scan(runs)
    print(f'{len(runs)} runs\n')
    for name, counter in counts.items():
        if not counter:
            continue
        print(name)
        for key, value in counter.most_common(a.top):
            print(f'  {value:6d}  {key}')
            if name.endswith('_gate') and example.get(key, key) != key:
                print(f'          e.g. {example[key][:88]}')
        print()
    if loops:
        print(f"loops\n  {loops['accepted']:6d}  accepted")


if __name__ == '__main__':
    main()
