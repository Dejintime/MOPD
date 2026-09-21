"""On-policy vLLM generation with an explicit trainer-to-engine weight sync."""

from dataclasses import asdict
import gc
import os

import torch

from .prompting import render_rollout_prompt


def release_ipc_cache(device):
    """Release completed CUDA IPC exports and return cached blocks to CUDA."""
    torch.cuda.synchronize(device)
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


class VLLMRollout:
    def __init__(self, cfg):
        if not cfg.get('student_device', '').startswith('cuda:'):
            raise ValueError('vLLM rollout requires a CUDA student')
        if cfg.get('enable_thinking') is not False:
            raise ValueError('This vLLM rollout requires explicit enable_thinking=false')
        settings = dict(cfg['vllm'])
        settings.setdefault('dtype', 'bfloat16')
        settings.setdefault('tensor_parallel_size', 1)
        settings.setdefault('max_model_len', cfg['max_prompt_tokens'] + cfg['max_new_tokens'])
        settings.setdefault('generation_config', 'vllm')
        settings.setdefault('enable_prefix_caching', False)
        if settings['max_model_len'] < cfg['max_prompt_tokens'] + cfg['max_new_tokens']:
            raise ValueError('vLLM context is shorter than the configured prompt and response budget')
        if settings['tensor_parallel_size'] != 1:
            raise ValueError('The colocated IPC weight sync currently supports one GPU')
        # The trainer has already initialized CUDA before the vLLM worker starts.
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        # Local CUDA IPC handles contain TensorMeta objects; vLLM's process
        # transport uses pickle for these trusted same-host messages.
        os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
        from vllm import LLM
        from vllm.config import WeightTransferConfig

        self.llm = LLM(model=cfg['student'], seed=cfg['seed'],
                       weight_transfer_config=WeightTransferConfig(backend='ipc'),
                       **settings)
        self.version = -1
        self._ipc_buffer = None
        self._ipc_handle = None

    def _ensure_ipc_buffer(self, student, buffer_bytes):
        """Allocate one packed buffer and reuse its IPC handle for every update."""
        if self._ipc_buffer is not None:
            if self._ipc_buffer.numel() != buffer_bytes:
                raise RuntimeError('Packed IPC buffer size changed during training')
            return
        from torch.multiprocessing.reductions import reduce_tensor

        device = next(student.parameters()).device
        with torch.cuda.device(device):
            self._ipc_buffer = torch.empty(
                buffer_bytes, dtype=torch.uint8, device=device)
            _, ipc_args = reduce_tensor(self._ipc_buffer)
            gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
            self._ipc_handle = {gpu_uuid: ipc_args}

    def _send_weights(self, student, buffer_bytes):
        """Synchronously copy checkpoint tensors through the reusable IPC buffer."""
        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCWeightTransferUpdateInfo)

        self._ensure_ipc_buffer(student, buffer_bytes)
        assert self._ipc_buffer is not None and self._ipc_handle is not None
        device = self._ipc_buffer.device
        names, dtype_names, shapes, tensor_sizes = [], [], [], []
        offset = 0

        def flush():
            nonlocal names, dtype_names, shapes, tensor_sizes, offset
            if not names:
                return
            torch.cuda.synchronize(device)
            update_info = IPCWeightTransferUpdateInfo(
                names=names, dtype_names=dtype_names, shapes=shapes,
                ipc_handles=self._ipc_handle, tensor_sizes=tensor_sizes,
                packed=True)
            self.llm.update_weights({'update_info': asdict(update_info)})
            names, dtype_names, shapes, tensor_sizes = [], [], [], []
            offset = 0

        for name, parameter in student.named_parameters():
            tensor = parameter.detach().contiguous()
            flat = tensor.view(torch.uint8).view(-1)
            size = flat.numel()
            if size > buffer_bytes:
                raise ValueError(
                    f"Tensor {name!r} needs {size} bytes, exceeding the "
                    f"{buffer_bytes}-byte packed IPC buffer")
            if offset and offset + size > buffer_bytes:
                flush()
            self._ipc_buffer[offset:offset + size].copy_(flat)
            names.append(name)
            dtype_names.append(str(tensor.dtype).split('.')[-1])
            shapes.append(list(tensor.shape))
            tensor_sizes.append(size)
            offset += size
        flush()

    def ensure_weights(self, student, version):
        """Never generate a post-update rollout with stale inference weights."""
        if self.version == version:
            return
        self.llm.start_weight_update(is_checkpoint_format=True)
        # Packed transfer bounds temporary GPU use while the trainer and
        # inference model share one device. A single embedding tensor is
        # roughly 742 MiB, so the buffer must be at least that large.
        largest = max(param.numel() * param.element_size()
                      for param in student.parameters())
        buffer_bytes = max(128 * 1024 * 1024,
                           ((largest + 32 * 1024 * 1024 - 1) // (32 * 1024 * 1024))
                           * 32 * 1024 * 1024)
        self._send_weights(student, buffer_bytes)
        self.llm.finish_weight_update()
        # Receiver-side clones and any other temporary blocks are no longer live.
        device = next(student.parameters()).device
        release_ipc_cache(device)
        self.version = version

    def generate_batch(self, rows, tokenizer, cfg, attempt):
        from vllm import SamplingParams

        prompts = []
        parameters = []
        for index, row in enumerate(rows):
            prompt = render_rollout_prompt(tokenizer, row['messages'], cfg,
                                           tokenize=False, add_generation_prompt=True)
            if not prompt.endswith('<think>\n\n</think>\n\n'):
                raise ValueError('Tokenizer did not render the Qwen3 no-thinking prefix')
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) > cfg['max_prompt_tokens']:
                raise ValueError('Prompt over budget; filter explicitly instead of truncating it')
            prompts.append(ids)
            parameters.append(SamplingParams(
                max_tokens=cfg['max_new_tokens'], temperature=cfg['temperature'],
                top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0,
                seed=cfg['seed'] + attempt * len(rows) + index, ignore_eos=False))
        outputs = self.llm.generate([{'prompt_token_ids': ids} for ids in prompts],
                                    parameters, use_tqdm=False)
        if len(outputs) != len(rows):
            raise RuntimeError('vLLM returned the wrong number of rollouts')
        rollouts = []
        for prompt_ids, result in zip(prompts, outputs):
            if list(result.prompt_token_ids) != prompt_ids or len(result.outputs) != 1:
                raise RuntimeError('vLLM rollout changed prompt tokens or returned multiple completions')
            completion = result.outputs[0]
            response_ids = list(completion.token_ids)
            if len(response_ids) > cfg['max_new_tokens']:
                raise RuntimeError('vLLM exceeded the configured response budget')
            if not response_ids or completion.finish_reason not in ('stop', 'length'):
                raise RuntimeError(f'Invalid vLLM completion: {completion.finish_reason}')
            ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            rollouts.append((ids, len(prompt_ids), response,
                             completion.finish_reason == 'length'))
        return rollouts

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
        device = self._ipc_buffer.device if self._ipc_buffer is not None else None
        self._ipc_handle = None
        self._ipc_buffer = None
        if device is not None:
            release_ipc_cache(device)
