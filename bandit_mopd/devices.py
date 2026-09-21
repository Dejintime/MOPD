"""Explicit placement for a CUDA student and sequential CPU/CUDA teachers."""
import os
import torch


def configure_devices(config):
    student = torch.device(config['student_device'])
    teacher = torch.device(config['teacher_device'])
    if student.type != 'cuda' or student.index is None:
        raise ValueError('student_device must be an explicit cuda:N device')
    if teacher.type not in ('cpu', 'cuda') or (teacher.type == 'cuda' and teacher.index is None):
        raise ValueError('teacher_device must be cpu or an explicit cuda:N device')
    if teacher.type == 'cpu' and teacher.index is not None:
        raise ValueError('Use teacher_device=cpu without an index')
    config.setdefault('teacher_dtype', 'float32' if teacher.type == 'cpu' else 'bfloat16')
    if config['teacher_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('teacher_dtype must be float32 or bfloat16')
    if teacher.type == 'cpu':
        # Do this before any CUDA queries/initialization, including mem_get_info.
        # Respect scheduler/user visibility: cuda:N indexes that existing list.
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU teacher isolation must be configured before CUDA initialization')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            selected = str(student.index)
        else:
            devices = [v.strip() for v in visible.split(',') if v.strip()]
            if student.index >= len(devices) or devices[student.index] == '-1':
                raise ValueError('student_device is outside CUDA_VISIBLE_DEVICES')
            selected = devices[student.index]
        os.environ['CUDA_VISIBLE_DEVICES'] = selected
        config['student_device'] = 'cuda:0'
        config['teacher_device'] = 'cpu'
    elif student == teacher:
        raise ValueError('CUDA teacher must use a different GPU from the student; use cpu for single-GPU runs')
    config['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')


def cuda_devices(config):
    """Only configured devices; never enumerate or query unrelated GPUs."""
    return sorted({torch.device(config[key]).index for key in ('student_device', 'teacher_device')
                   if torch.device(config[key]).type == 'cuda'})


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def release_cache(device):
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def peak_gpu_memory(config):
    return {str(i): torch.cuda.max_memory_allocated(i) for i in cuda_devices(config)}
