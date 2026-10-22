"""Exercise actual launcher validation and discovered per-rank mapping without SSH/GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')


class LauncherTransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        for name in ('config.sh', 'nodes.sh'):
            shutil.copyfile(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        shutil.copyfile(ROOT / 'start.sh', self.root / 'start.sh')
        forbidden = self.root / 'forbidden'
        forbidden.mkdir()
        self.marker = self.root / 'external-called'
        for name in ('ssh', 'docker', 'rsync', 'sudo'):
            path = forbidden / name
            path.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> "$EXTERNAL_MARKER"\nexit 97\n')
            path.chmod(0o755)
        self.env = {'PATH': str(forbidden) + os.pathsep + os.environ['PATH'],
                    'HOME': str(self.root), 'EXTERNAL_MARKER': str(self.marker),
                    'WORKER': 'rank1', 'WORKER2': 'rank2', 'WORKER3': 'rank3',
                    'ENGINE': 'zig', 'TP': '4', 'SPLIT': '0', 'COMM': 'nccl',
                    'FABRIC_EXPECT': 'ring', 'MASTER_ADDR': '192.0.2.1',
                    'KERNEL_CACHE': str(self.root / 'kernel-cache'), 'STATE_DIR': str(self.root / 'state')}

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, mode='check-config', **settings):
        env = self.env | settings
        result = subprocess.run([BASH, str(self.root / 'start.sh'), *([mode] if mode else [])],
                                env=env, text=True, capture_output=True, timeout=10)
        self.assertFalse(self.marker.exists(), result.stdout + result.stderr)
        self.assertFalse((self.root / 'kernel-cache').exists())
        self.assertFalse((self.root / 'state').exists())
        return result

    def test_default_nccl_and_legacy_tp2_tp3_are_preserved(self):
        for settings, roce in (({}, '0'), ({'TP': '2', 'COMM': 'roce'}, '1'),
                               ({'TP': '3', 'COMM': 'roce'}, '1')):
            with self.subTest(settings=settings):
                result = self.launch(**settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('TF_GLM_TRANSPORT=legacy TF_GLM_ROCE=' + roce, result.stdout)

    def test_explicit_relay_and_neighbor_resolve_the_legacy_switch(self):
        for mode, comm, roce in (('roce', 'roce', '1'), ('neighbor', 'nccl', '0')):
            result = self.launch(TF_GLM_TRANSPORT=mode, COMM=comm,
                                 TF_GLM_NEIGHBOR_GID_INDEX='3')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('TF_GLM_TRANSPORT=' + mode + ' TF_GLM_ROCE=' + roce, result.stdout)

    def test_invalid_scopes_fail_before_external_actions(self):
        cases = [({'COMM': 'roce'}, 'explicit research'),
                 ({'TF_GLM_TRANSPORT': ''}, 'cannot be empty'),
                 ({'TF_GLM_ROCE_MEMORY': ''}, 'allocated or registered'),
                 ({'TF_GLM_ROCE_REQUIRED': ''}, '0 or 1'),
                 ({'TF_GLM_TRANSPORT': 'auto'}, 'TF_GLM_TRANSPORT is'),
                 ({'TF_GLM_TRANSPORT': 'roce', 'COMM': 'roce', 'ENGINE': 'python'}, 'ENGINE=zig TP=4'),
                 ({'TF_GLM_TRANSPORT': 'roce', 'COMM': 'roce', 'TP': '3'}, 'ENGINE=zig TP=4'),
                 ({'TF_GLM_TRANSPORT': 'roce', 'COMM': 'roce', 'FABRIC_EXPECT': 'mesh'}, 'FABRIC_EXPECT=ring SPLIT=0'),
                 ({'TF_GLM_TRANSPORT': 'roce', 'COMM': 'roce', 'SPLIT': '1'}, 'FABRIC_EXPECT=ring SPLIT=0'),
                 ({'TF_GLM_TRANSPORT': 'roce'}, 'needs COMM=roce'),
                 ({'TF_GLM_TRANSPORT': 'neighbor', 'COMM': 'roce'}, 'needs COMM=nccl'),
                 ({'TF_GLM_TRANSPORT': 'neighbor'}, 'explicit TF_GLM_NEIGHBOR_GID_INDEX=3'),
                 ({'TF_GLM_TRANSPORT': 'neighbor', 'TF_GLM_NEIGHBOR_GID_INDEX': '3', 'TF_GLM_NEIGHBOR_FIRST': 'a,b'}, 'derived per rank'),
                 ({'TF_GLM_ROCE_MEMORY': 'registered'}, 'NCCL-only'),
                 ({'TF_GLM_ROCE_REQUIRED': '1'}, 'NCCL-only'),
                 ({'TF_GLM_ROCE_MEMORY': 'automatic'}, 'allocated or registered')]
        for settings, message in cases:
            with self.subTest(settings=settings):
                result = self.launch(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_unsupported_neighbor_gids_block_start_and_restart_before_mutation(self):
        for mode in ('check-config', '', 'restart'):
            for gid in ('0', '2', '4', '5', '03', '256', 'invalid', ''):
                with self.subTest(mode=mode, gid=gid):
                    result = self.launch(mode=mode, TF_GLM_TRANSPORT='neighbor',
                                         TF_GLM_NEIGHBOR_GID_INDEX=gid, PREPARE='1')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('TF_GLM_NEIGHBOR_GID_INDEX=3', result.stderr)

    def test_unexported_local_selector_and_memory_reach_rank_environment(self):
        (self.root / 'scripts/local.sh').write_text(
            'COMM=roce\nTF_GLM_TRANSPORT=roce\nTF_GLM_ROCE_MEMORY=registered\n')
        self.env.pop('COMM')
        receipt = self.root / 'exported-settings'
        hook = self.root / 'bash-env'
        hook.write_text("trap 'declare -p TF_GLM_TRANSPORT TF_GLM_ROCE_MEMORY > \"$EXPORT_RECEIPT\"' EXIT\n")
        result = self.launch(BASH_ENV=str(hook), EXPORT_RECEIPT=str(receipt))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('declare -x TF_GLM_TRANSPORT="roce"', receipt.read_text())
        self.assertIn('declare -x TF_GLM_ROCE_MEMORY="registered"', receipt.read_text())

    def test_invalid_selection_blocks_real_start_and_restart_before_mutation(self):
        for mode in ('', 'restart'):
            result = self.launch(mode=mode, TF_GLM_TRANSPORT='roce', COMM='roce', TP='3', PREPARE='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ENGINE=zig TP=4', result.stderr)

    def mapping(self, edge_override=''):
        script = '''set -euo pipefail
source "$1/scripts/config.sh"
source "$1/scripts/nodes.sh"
FABRIC=ring
for r in 0 1 2 3; do
  NODE_DEV[r]="net$r"; NODE_HCAS[r]="a$r,b$r,c$r,d$r"; NODE_GID[r]=3
  TOWARD["$r $((r ^ 1))"]="a$r,b$r"
  TOWARD["$r $((r ^ 3))"]="c$r,d$r"
done
''' + edge_override + '''
check_neighbor_links
for r in 0 1 2 3; do rank_nccl_env "$r"; done
'''
        result = subprocess.run([BASH, '-c', script, 'mapping-test', str(self.root)],
                                env=self.env | {'TF_GLM_TRANSPORT': 'neighbor'},
                                text=True, capture_output=True, timeout=10)
        self.assertFalse(self.marker.exists())
        return result

    def test_heterogeneous_rank_names_use_the_discovered_xor_edges(self):
        result = self.mapping()
        self.assertEqual(result.returncode, 0, result.stderr)
        for rank in range(4):
            self.assertIn(f'TF_GLM_NEIGHBOR_FIRST=a{rank},b{rank}', result.stdout)
            self.assertIn(f'TF_GLM_NEIGHBOR_SECOND=c{rank},d{rank}', result.stdout)
        self.assertNotIn('TF_GLM_ROCE=1', result.stdout)

    def test_neighbor_discovery_refuses_one_rail_overlap_and_invalid_names(self):
        for override, message in [('TOWARD["2 3"]=a2', 'exactly two distinct'),
                                  ('TOWARD["2 1"]=a2,d2', 'overlapping'),
                                  ('TOWARD["2 1"]="bad rail,d2"', 'invalid discovered')]:
            with self.subTest(override=override):
                result = self.mapping(override)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
