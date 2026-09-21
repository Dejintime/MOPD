"""Apply rollout instructions consistently without changing dataset identities."""


def rollout_messages(messages, cfg):
    result = [dict(message) for message in messages]
    instruction = cfg.get('rollout_system_prompt', '')
    if not isinstance(instruction, str):
        raise ValueError('rollout_system_prompt must be text')
    if instruction.strip():
        if result and result[0]['role'] == 'system':
            result[0]['content'] = result[0]['content'].rstrip() + '\n\n' + instruction
        else:
            result.insert(0, {'role': 'system', 'content': instruction})
    return result


def render_rollout_prompt(tokenizer, messages, cfg, **kwargs):
    if 'enable_thinking' in cfg:
        kwargs['enable_thinking'] = cfg['enable_thinking']
    return tokenizer.apply_chat_template(rollout_messages(messages, cfg), **kwargs)
