import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prepare_eval',Path(__file__).parents[1]/'scripts/experiments/prepare_m2rl_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_aime_answer_preserved_without_brittle_slicing():
    row={'problem':'Question','solution':r'\boxed{007}','id':1}
    result=module.math_record('aime24',row,0)
    assert result['answer']=='007'
    assert result['messages'][0]['content'].endswith('Question')
    with pytest.raises(ValueError):
        module.math_record('aime24',{**row,'solution':r'\boxed{1000}'},0)


def test_gpqa_option_mapping_is_stable_and_correct():
    row={'Question':'Q?','Correct Answer':' Correct ',
         'Incorrect Answer 1':'One','Incorrect Answer 2':'Two','Incorrect Answer 3':'Three'}
    a=module.gpqa_record(row,0)
    b=module.gpqa_record(row,0)
    assert a==b
    index='ABCD'.index(a['answer'])
    assert a['metadata']['choices'][index]=='Correct'
    assert f"{a['answer']}) Correct" in a['messages'][0]['content']
    assert a['usage']=='final_evaluation_only'


def test_lcb_keeps_hidden_tests_opaque():
    row={'question_content':'Q','question_id':'42','private_test_cases':'opaque_encoded_data',
         'public_test_cases':'[]','starter_code':'def solve(): pass'}
    result=module.lcb_record(row,6)
    assert result['metadata']['private_test_cases']=='opaque_encoded_data'
    assert 'opaque_encoded_data' not in result['messages'][0]['content']
    assert result['metadata']['incremental_version']==6


def test_overlap_screen_requires_complete_question_and_preserves_operators():
    spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/experiments/audit_dataset_overlap.py')
    audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    text='For the following integers, compute the final value of a + b and show the complete reasoning.'
    matcher=audit.QuestionMatcher([{'benchmark':'math','id':'1','question':text}])
    assert matcher.match([{'content':'Solve this:\n'+text+'\nEnd with an answer.'}])==[('math','1')]
    assert matcher.match([{'content':text.replace('a + b','a - b')}])==[]
    assert matcher.match([{'content':text[:70]+'different remainder'}])==[]
