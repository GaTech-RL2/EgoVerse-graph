"""Synthetic retained-file/CSV semantics only; no checkpoint contents are proved."""
import csv
from pathlib import Path
import pytest
from scripts.train.joint import imagenet


def checkpoints(tmp_path):
    root=tmp_path/'run'; directory=root/'checkpoints'; directory.mkdir(parents=True)
    result={}
    for step in range(20000,300001,20000):
        path=directory/f'epoch_epoch=0000-step_step={step:09d}.ckpt';path.write_bytes(b'fixture-not-a-checkpoint')
        result[step]=path
    (directory/'last.ckpt').symlink_to(result[300000].name)
    return root,result


def history(path,steps,values=None):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step',imagenet.METRIC]);writer.writeheader()
        for i,step in enumerate(steps):writer.writerow({'step':step,imagenet.METRIC:.1 if values is None else values[i]})
    return path


def test_exact_local_scheduled_inventory_and_actual_lightning_default_names(tmp_path):
    root,expected=checkpoints(tmp_path)
    found,last=imagenet.retained_milestones(root)
    assert found==expected and last.resolve()==expected[300000]


@pytest.mark.parametrize('damage',['missing','empty','wrong_name','extra','symlink','duplicate','wrong_last','regular_last'])
def test_retention_metadata_vetoes(tmp_path,damage):
    root,paths=checkpoints(tmp_path); directory=root/'checkpoints';first=paths[20000]
    if damage=='missing':first.unlink()
    if damage=='empty':first.write_bytes(b'')
    if damage=='wrong_name':first.rename(directory/'arbitrary000020000.ckpt')
    if damage=='extra':(directory/'epoch_epoch=0000-step_step=000001000.ckpt').write_bytes(b'extra')
    if damage=='symlink':first.unlink();first.symlink_to(paths[40000].name)
    if damage=='duplicate':(directory/'epoch_epoch=0001-step_step=000020000.ckpt').write_bytes(b'duplicate')
    if damage in ('wrong_last','regular_last'):
        (directory/'last.ckpt').unlink()
        if damage=='wrong_last':(directory/'last.ckpt').symlink_to(paths[280000].name)
        else:(directory/'last.ckpt').write_bytes(b'not-link')
    with pytest.raises(ValueError):imagenet.retained_milestones(root)


@pytest.mark.parametrize('offset',[0,-1])
def test_actual_lightning_step_conventions_are_supported(tmp_path,offset):
    steps=list(range(20000,300001,20000));path=history(tmp_path/'metrics.csv',[s+offset for s in steps])
    assert len(imagenet.validation_history([path],steps))==15


@pytest.mark.parametrize('damage',['duplicate_one_log','duplicate_across_logs','missing','extra','nan','inf','fractional','negative','unexpected_metric'])
def test_history_vetoes_before_claiming_scheduled_validation(tmp_path,damage):
    milestones=list(range(20000,300001,20000));steps=list(milestones);paths=[tmp_path/'metrics.csv'];values=[.1]*15
    if damage=='duplicate_one_log':steps.append(steps[0]);values.append(.2)
    if damage=='missing':steps.pop();values.pop()
    if damage=='extra':steps.append(1234);values.append(.1)
    if damage=='nan':values[0]='nan'
    if damage=='inf':values[0]='inf'
    if damage=='fractional':steps[0]=20000.5
    if damage=='negative':steps[0]=-1
    history(paths[0],steps,values)
    if damage=='duplicate_across_logs':paths.append(history(tmp_path/'other.csv',[milestones[0]]))
    if damage=='unexpected_metric':paths[0].write_text(paths[0].read_text().replace(imagenet.METRIC,'val/imagenet/other'))
    with pytest.raises(ValueError):imagenet.validation_history(paths,milestones)
