#!/usr/bin/env python3
"""Reconstruct the pinned engine and verify research packaging without network/GPU access."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args],
                                   env=os.environ | {'GIT_NO_LAZY_FETCH': '1'})


def check(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-repo', type=Path, required=True,
                        help='local TensorFold object database containing the pinned upstream commit')
    parser.add_argument('--production-repo', type=Path, required=True,
                        help='local frozen baseline and tested production source repository')
    parser.add_argument('--probe-repo', type=Path, required=True,
                        help='local source repository containing probe04')
    parser.add_argument('--neighbor-repo', type=Path,
                        help='local imported-protocol object database (defaults to upstream-repo)')
    parser.add_argument('--tree-dir', type=Path,
                        help='retain reconstruction at a new directory for optional host-only builds')
    args = parser.parse_args()
    if args.tree_dir:
        args.tree_dir.mkdir(parents=True, exist_ok=False)
    meta = json.loads((ROOT / 'zig/TRANSPORT_SOURCE.json').read_text())
    base = meta['reconstructed_baseline']['commit']
    prod = meta['production']['commit']
    probe = meta['probe']['commit']
    probe_path = meta['probe']['path']
    paths = set(meta['production']['source_paths'])
    omitted = meta['packaging']['omitted_operations_note']
    changed = set(git(args.production_repo, 'diff', '--name-only', base, prod).decode().splitlines())
    check(changed == paths | {omitted}, 'Production changed-path set differs from provenance')
    changed_probe = git(args.probe_repo, 'diff', '--name-only', prod, probe).decode().splitlines()
    check(changed_probe == [probe_path], 'Probe revisions contain changes outside the probe source')
    combined = git(args.probe_repo, 'diff', '--binary', '--no-renames', prod, probe, '--', probe_path)
    check(hashlib.sha256(combined).hexdigest() == meta['probe']['combined_patch_sha256'], 'Combined probe diff hash differs')
    check(json.loads(git(args.production_repo, 'show', prod + ':glm-neighbor-source.json')) == meta['neighbor_protocol'],
          'Imported-protocol provenance differs from tested production')
    protocol = meta['neighbor_protocol']
    for path, expected_hash in protocol['sha256'].items():
        original_source = git(args.neighbor_repo or args.upstream_repo, 'show', protocol['commit'] + ':' + path)
        check(hashlib.sha256(original_source).hexdigest() == expected_hash, 'Imported-protocol upstream source hash differs: ' + path)
    patch = ROOT / meta['packaging']['patch']
    check(hashlib.sha256(patch.read_bytes()).hexdigest() == meta['packaging']['sha256'], 'Research patch hash differs')
    context = nullcontext(str(args.tree_dir)) if args.tree_dir else tempfile.TemporaryDirectory(prefix='glm-transport-source-')
    with context as directory:
        tree = Path(directory)
        archive = git(args.upstream_repo, 'archive', meta['tensorfold']['commit'])
        subprocess.run(['tar', '-xf', '-', '-C', str(tree)], input=archive, check=True)
        git(tree, 'init', '-q')
        patches = sorted((ROOT / 'zig/patches').glob('*.patch'))
        check(len(patches) == 14 and patches[-1] == patch, 'Expected exactly the pinned 13 patches plus research patch 0014')
        for item in patches[:-1]:
            git(tree, 'apply', '--check', str(item))
            git(tree, 'apply', str(item))
        git(tree, 'add', '-f', '-A')
        baseline_tree = git(tree, 'write-tree').decode().strip()
        frozen = meta['reconstructed_baseline']
        check(baseline_tree == frozen['upstream_plus_patches_tree'], 'First 13 patches do not reproduce the pinned reconstructed tree')
        check(git(args.production_repo, 'rev-parse', base + '^{tree}').decode().strip() == frozen['tree'], 'Frozen baseline identity differs')
        def entries(repo, ref):
            result = {}
            for line in git(repo, 'ls-tree', '-r', ref).decode().splitlines():
                info, path = line.split('\t', 1)
                mode, _, blob = info.split()
                result[path] = (mode, blob)
            return result
        original = entries(args.production_repo, base)
        rebuilt = entries(tree, baseline_tree)
        check(original.keys() == rebuilt.keys(), 'Baseline reconstructed paths differ')
        exceptions = {}
        for path in original:
            check(original[path][1] == rebuilt[path][1], 'Baseline source bytes differ: ' + path)
            if original[path][0] != rebuilt[path][0]:
                exceptions[path] = {'frozen_baseline': original[path][0], 'upstream_plus_patches': rebuilt[path][0]}
        check(exceptions == frozen['mode_only_differences'], 'Baseline executable-mode exceptions differ')
        git(tree, 'apply', '--check', str(patch))
        git(tree, 'apply', str(patch))
        git(tree, 'add', '-f', '-A')
        actual_changes = set(git(tree, 'diff', '--cached', '--name-only', baseline_tree).decode().splitlines())
        check(actual_changes == paths, 'Research patch changes an unexpected path')
        for path in sorted(paths):
            expected_prod = git(args.production_repo, 'show', prod + ':' + path)
            check(hashlib.sha256(expected_prod).hexdigest() == meta['production']['sha256'][path], 'Production source hash differs: ' + path)
            expected = git(args.probe_repo, 'show', probe + ':' + path) if path == probe_path else expected_prod
            check((tree / path).read_bytes() == expected, 'Reconstructed source differs: ' + path)
        check(hashlib.sha256((tree / probe_path).read_bytes()).hexdigest() == meta['probe']['sha256'], 'Probe04 source hash differs')
        result_tree = git(tree, 'write-tree').decode().strip()
    print(json.dumps({'pass': True, 'baseline_tree': baseline_tree,
                      'research_tree': result_tree, 'source_paths': len(paths),
                      'patch_sha256': meta['packaging']['sha256'],
                      'production_byte_identical': True, 'baseline_blob_identical': True,
                      'baseline_mode_exceptions': sorted(exceptions), 'probe_source': probe,
                      'network': False, 'NVCC': False, 'GPU': False}, indent=2))


if __name__ == '__main__':
    main()
