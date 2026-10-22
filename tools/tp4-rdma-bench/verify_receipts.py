#!/usr/bin/env python3
"""Verify the exported receipts without contacting a server or loading credentials."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(manifest, directory):
    root = directory.resolve(strict=True)
    rows = manifest['files']
    seen = set()
    total = 0
    for row in rows:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) != row['path']:
            raise ValueError('Noncanonical receipt path')
        if row['path'] in seen:
            raise ValueError('Duplicate receipt path')
        seen.add(row['path'])
        path = root.joinpath(*relative.parts)
        if not path.resolve(strict=True).is_relative_to(root) or path.is_symlink():
            raise ValueError('Receipt escapes root or is a symlink')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Receipt size or hash mismatch: ' + row['path'])
        total += len(data)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != seen:
        raise ValueError('Manifest and receipt file sets differ')
    if len(rows) != manifest['file_count'] or total != manifest['total_bytes']:
        raise ValueError('Manifest totals disagree')
    return {'passed': True, 'files': len(rows), 'bytes': total}


def main():
    default = Path(__file__).resolve().parents[2] / 'docs/tp4-rdma-research'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=default / 'receipts-manifest.json')
    parser.add_argument('--receipts', type=Path, default=default / 'receipts')
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.manifest.read_text()), args.receipts)))


if __name__ == '__main__':
    main()
