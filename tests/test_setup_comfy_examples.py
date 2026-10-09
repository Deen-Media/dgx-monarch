"""ComfyUI example deletions are the only permitted dirty setup checkout."""
import json
import os
import subprocess

import pytest

from dgx_monarch.cli import setup_config_io, setup_probe

EXAMPLES = ('input/example.png', 'output/_output_images_will_be_put_here')


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def checkout(tmp_path):
    for name in (*EXAMPLES, 'comfy/sd.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('tracked\n')
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'add', '.')
    git(tmp_path, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'initial')
    return tmp_path


def probe(root):
    result = subprocess.run(['bash', '-s'], input=setup_probe.build_probe_script('/usr/bin/python3', str(root), ()), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    parsed = setup_probe.parse_probe_output(result.stdout, 1, 0)
    assert parsed.reachable and parsed.failure is None
    return parsed


def readiness(parsed):
    return setup_config_io.setup_readiness(probes=(parsed,), expected_gpus=(1,), artifacts=(), fabric_profile='single-node', install_service=False, start_workers=False, verify=False, transport_security='trusted_fabric')


@pytest.mark.parametrize('missing', [(), EXAMPLES[:1], EXAMPLES[1:], EXAMPLES])
def test_only_unstaged_regular_example_deletions_are_accepted(checkout, missing):
    for name in missing:
        (checkout / name).unlink()
    before = git(checkout, 'status', '--porcelain=v1', '-z')
    parsed = probe(checkout)
    assert parsed.comfy_dirty is bool(missing)
    assert parsed.comfy_only_missing_examples is bool(missing)
    blockers, warnings = readiness(parsed)
    assert not any('comfy_' in reason for reason in blockers)
    assert ('host_1_comfy_missing_examples_preserved' in warnings) is bool(missing)
    assert git(checkout, 'status', '--porcelain=v1', '-z') == before
    assert all(not os.path.lexists(checkout / name) for name in missing)
    assert parsed.as_dict()['comfy_dirty'] is bool(missing)


@pytest.mark.parametrize('change', ['runtime', 'untracked', 'untracked_newline', 'staged_delete', 'staged_edit', 'rename', 'example_edit', 'symlink', 'head_symlink', 'assume', 'skip', 'index_mode'])
def test_every_other_checkout_change_remains_blocked(checkout, change):
    example = checkout / EXAMPLES[0]
    if change == 'head_symlink':
        example.unlink()
        example.symlink_to('../comfy/sd.py')
        git(checkout, 'add', EXAMPLES[0])
        git(checkout, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'symlink')
    example.unlink()
    if change == 'runtime':
        (checkout / 'comfy/sd.py').write_text('modified\n')
    elif change in ('untracked', 'untracked_newline'):
        (checkout / ('unknown\n D input/example.png' if change.endswith('newline') else 'unknown')).parent.mkdir(parents=True, exist_ok=True)
        (checkout / ('unknown\n D input/example.png' if change.endswith('newline') else 'unknown')).write_text('untracked')
    elif change == 'staged_delete':
        git(checkout, 'add', EXAMPLES[0])
    elif change == 'staged_edit':
        (checkout / 'comfy/sd.py').write_text('modified\n')
        git(checkout, 'add', 'comfy/sd.py')
    elif change == 'rename':
        git(checkout, 'mv', EXAMPLES[1], 'output/renamed')
    elif change == 'example_edit':
        (checkout / EXAMPLES[1]).write_text('modified\n')
    elif change == 'symlink':
        example.symlink_to('../comfy/sd.py')
    elif change in ('assume', 'skip'):
        git(checkout, 'update-index', '--assume-unchanged' if change == 'assume' else '--skip-worktree', 'comfy/sd.py')
    elif change == 'index_mode':
        example.write_text('tracked\n')
        git(checkout, 'update-index', '--chmod=+x', EXAMPLES[0])
        example.unlink()
    parsed = probe(checkout)
    assert parsed.comfy_dirty is True
    assert parsed.comfy_only_missing_examples is False
    assert 'host_1_comfy_dirty' in readiness(parsed)[0]
    assert not readiness(parsed)[1]


def test_older_dirty_payload_has_no_exception(checkout):
    (checkout / EXAMPLES[0]).unlink()
    script = setup_probe.build_probe_script('/usr/bin/python3', str(checkout), ())
    completed = subprocess.run(['bash', '-s'], input=script, capture_output=True, text=True, timeout=30, check=True)
    line = next(line for line in completed.stdout.splitlines() if line.startswith('DGXM_SETUP_PROBE='))
    payload = json.loads(line.split('=', 1)[1])
    del payload['comfy_only_missing_examples']
    parsed = setup_probe.parse_probe_output('DGXM_SETUP_PROBE=' + json.dumps(payload), 1, 0)
    assert parsed.comfy_dirty is True and parsed.comfy_only_missing_examples is None
    assert 'host_1_comfy_dirty' in readiness(parsed)[0]
