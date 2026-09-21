import json
from pathlib import Path
import pytest
import torch

from bandit_mopd.checkpoints import configure_checkpointing, disk_reserve_bytes, should_save, save_checkpoint, link_final


def config(**kwargs):
    return {'steps': 100, 'save_student': True, 'save_optimizer': True, **kwargs}


def test_default_interval_is_optimizer_steps_50_and_100():
    cfg = config()
    assert configure_checkpointing(cfg) == 50
    assert [i for i in range(1, 101) if should_save(i, cfg)] == [50, 100]
    assert disk_reserve_bytes(cfg) == 130*1024**3


def test_nonmultiple_final_step_saved_once_and_final_only_option():
    cfg = config(steps=125)
    assert [i for i in range(1, 126) if should_save(i, cfg)] == [50, 100, 125]
    cfg['save_steps'] = 0
    assert [i for i in range(1, 126) if should_save(i, cfg)] == [125]
    cfg.update(save_student=False, save_optimizer=False)
    assert not should_save(125, cfg) and disk_reserve_bytes(cfg) == 0


@pytest.mark.parametrize('value', [-1, True, 1.5, '50'])
def test_invalid_interval_rejected(value):
    with pytest.raises(ValueError): configure_checkpointing(config(save_steps=value))


class Student:
    def __init__(self): self.fail = False
    def save_pretrained(self, path, **kwargs):
        path.mkdir()
        (path/'model.safetensors').write_bytes(b'fixture weights')
        if self.fail:
            raise OSError('simulated write failure')


class Tokenizer:
    def save_pretrained(self, path):
        (path/'tokenizer.json').write_text('{}')


class State:
    def state_dict(self): return {'step': 50}


def test_failed_save_preserves_previous_checkpoint_and_latest_pointer(tmp_path):
    student = Student()
    first = save_checkpoint(tmp_path, student, Tokenizer(), State(), State(), config(),
                            {'optimizer_step': 50, 'train_samples_seen': 200})
    assert json.loads((first/'trainer_state.json').read_text())['status'] == 'complete'
    assert torch.load(first/'optimizer.pt', weights_only=True) == {'step': 50}
    latest_before = (tmp_path/'latest_checkpoint.json').read_bytes()
    student.fail = True
    with pytest.raises(OSError, match='simulated'):
        save_checkpoint(tmp_path, student, Tokenizer(), State(), State(), config(), {'optimizer_step': 100})
    assert not (tmp_path/'checkpoint-100').exists()
    assert not list(tmp_path.glob('.checkpoint-*'))
    assert (tmp_path/'latest_checkpoint.json').read_bytes() == latest_before
    assert (first/'student/model.safetensors').read_bytes() == b'fixture weights'
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, student, Tokenizer(), State(), State(), config(), {'optimizer_step': 50})


def test_final_links_reuse_complete_weights_and_do_not_overwrite(tmp_path):
    checkpoint = save_checkpoint(tmp_path, Student(), Tokenizer(), State(), State(), config(), {'optimizer_step': 100})
    link_final(tmp_path, checkpoint, config())
    assert (tmp_path/'student').is_symlink()
    assert (tmp_path/'student').resolve() == checkpoint/'student'
    assert (tmp_path/'optimizer.pt').resolve() == checkpoint/'optimizer.pt'
    assert len(list(tmp_path.glob('checkpoint-*'))) == 1
    with pytest.raises(FileExistsError): link_final(tmp_path, checkpoint, config())
