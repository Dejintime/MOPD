"""Effective-batch sizing and padded, response-only student forwards."""
import torch


def batch_settings(config):
    for key in ('micro_batch_size', 'gradient_accumulation_steps'):
        value = config.setdefault(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    return config['micro_batch_size'] * config['gradient_accumulation_steps']


def response_log_probs_batch(model, samples, device, pad_token_id=0):
    """Right-pad full trajectories, excluding prompts/padding from the loss.

    Each result is [response_tokens, vocab]. Position IDs preserve single-sample
    semantics for variable prompt/response lengths. Only one microbatch's graphs
    remain live; the caller backpropagates before preparing the next microbatch.
    """
    if not samples:
        raise ValueError('Empty microbatch')
    lengths = [s['ids'].shape[1] for s in samples]
    prompts = [s['prompt_length'] for s in samples]
    if any(not 1 <= p < n for p, n in zip(prompts, lengths)):
        raise ValueError('Every trajectory needs prompt and response tokens')
    width = max(lengths)
    ids = torch.full((len(samples), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        ids[i, :length] = sample['ids'][0].to(device)
        mask[i, :length] = 1
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    offset = min(prompts)-1
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                use_cache=False, logits_to_keep=width-offset)
    return [out.logits[i, p-1-offset:n-1-offset].float().log_softmax(-1)
            for i, (p, n) in enumerate(zip(prompts, lengths))]
