"""Trusted text checkers. Run through the bounded worker in training."""
import hashlib
import random
import re


def final_response(text):
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text:
        return ''
    return text.strip()


def answer_span(text):
    """Last explicit answer/boxed answer; never credit an intermediate result."""
    spans = []
    for match in re.finditer(r'\\boxed\{', text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                spans.append((match.start(), text[start:end]))
                break
    for match in re.finditer(r'(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*([^\n]+)', text, re.I):
        spans.append((match.start(), match.group(1).strip()))
    if spans:
        return max(spans, key=lambda x: x[0])[1]
    return text if '\n' not in text and len(text) <= 200 else ''


def science(row, response):
    text = final_response(response)
    answer = str(row['answer']).strip().upper()
    # Honor dataset templates, but reject ambiguous lists such as Answer: A/B.
    pattern = (row['metadata'].get('template_metadata') or {}).get('output_regex')
    matches = list(re.finditer(pattern, text)) if pattern else []
    if matches:
        match = matches[-1]
        prediction = next((s for s in match.groups() if s is not None), match.group()).strip()
        tail = text[match.end():].lstrip()
        group_end = match.end(next((i for i, value in enumerate(match.groups(), 1) if value is not None), 0))
        if (tail.startswith(('/', ',', '|')) or re.match(r'(?:or|and)\b', tail, re.I)
                or (group_end < len(text) and text[group_end].isalnum())):
            prediction = ''
    else:
        prediction = answer_span(text)
    correct = prediction.strip().upper() == answer
    return {'score': float(correct), 'correct': correct, 'prediction': prediction}


def math(row, response):
    from math_verify import parse, verify
    gold = str(row['answer']).strip()
    prediction = answer_span(final_response(response))
    if not prediction:
        return {'score': 0.0, 'correct': False, 'reason': 'missing_final_answer'}
    # Never execute model text; Math-Verify handles LaTeX/expression equivalence.
    # Both parsers have their own time limits in addition to the worker deadline.
    parsed_gold = parse('$' + gold.strip('$') + '$', fallback_mode='no_fallback', parsing_timeout=3)
    parsed_pred = parse('$' + prediction.strip('$') + '$', fallback_mode='no_fallback', parsing_timeout=3)
    if parsed_gold and parsed_pred:
        correct = bool(verify(parsed_gold, parsed_pred, timeout_seconds=3))
        method = 'math_verify'
    else:
        correct = re.sub(r'\s+', '', gold) == re.sub(r'\s+', '', prediction)
        method = 'exact_text_fallback'
    return {'score': float(correct), 'correct': correct, 'prediction': prediction, 'method': method}


def build_instructions(row):
    from third_party.ifevalg.instructions_registry import INSTRUCTION_DICT
    instructions = []
    state = random.getstate()
    random.seed(int(hashlib.sha256(row['id'].encode()).hexdigest()[:16], 16))
    try:
        for name, kwargs in zip(row['metadata']['instruction_id_list'], row['metadata']['kwargs']):
            instruction = INSTRUCTION_DICT[name](name)
            instruction.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
            args = instruction.get_instruction_args()
            if args and 'prompt' in args:
                prompt = '\n'.join(m['content'] for m in row['messages'] if m['role'] == 'user')
                instruction.build_description(prompt=prompt)
            instructions.append(instruction)
    finally:
        random.setstate(state)
    return instructions


def instruction_following(row, response):
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    text = final_response(response)
    checks = [bool(text) and bool(i.check_following(text)) for i in build_instructions(row)]
    return {'score': float(all(checks)), 'correct': all(checks),
            'instruction_ids': row['metadata']['instruction_id_list'], 'instruction_passed': checks,
            'instruction_fraction': sum(checks)/len(checks)}
