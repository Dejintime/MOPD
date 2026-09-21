"""Separate bounded process for trusted symbolic/IF checkers, not generated code."""
import contextlib
import json
import sys
from . import text


def main():
    payload = json.load(sys.stdin)
    try:
        checker = {'math': text.math, 'science': text.science, 'if': text.instruction_following}[payload['row']['domain']]
        with contextlib.redirect_stdout(sys.stderr):
            result = checker(payload['row'], payload['response'])
        print(json.dumps({'result': result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
        sys.exit(2)


if __name__ == '__main__':
    main()
