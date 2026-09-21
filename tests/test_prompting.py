import copy
import pytest
from bandit_mopd.prompting import rollout_messages, render_rollout_prompt


def test_adds_system_instruction_without_mutating_original_messages():
    messages=[{'role':'user','content':'Solve this problem step by step.'}]
    original=copy.deepcopy(messages)
    result=rollout_messages(messages,{'rollout_system_prompt':'Give the final result directly.'})
    assert messages==original
    assert result==[{'role':'system','content':'Give the final result directly.'},*original]


def test_preserves_existing_system_requirements():
    messages=[{'role':'system','content':'Answer in French.'},{'role':'user','content':'Question'}]
    original=copy.deepcopy(messages)
    result=rollout_messages(messages,{'rollout_system_prompt':'Be concise.'})
    assert result[0]['content']=='Answer in French.\n\nBe concise.'
    assert result[1]==original[1] and messages==original


def test_absent_settings_preserve_legacy_template_call():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return messages,kwargs
    messages=[{'role':'user','content':'Question'}]
    result,kwargs=render_rollout_prompt(Tokenizer(),messages,{},tokenize=True,add_generation_prompt=True)
    assert result==messages
    assert kwargs=={'tokenize':True,'add_generation_prompt':True}


def test_thinking_flag_and_added_prompt_reach_token_length_and_render_calls():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['enable_thinking'] is False
            text='|'.join(m['content'] for m in messages)
            return list(text) if kwargs['tokenize'] else text
    cfg={'rollout_system_prompt':'Be concise.','enable_thinking':False}
    messages=[{'role':'user','content':'Question'}]
    rendered=render_rollout_prompt(Tokenizer(),messages,cfg,tokenize=False)
    length=len(render_rollout_prompt(Tokenizer(),messages,cfg,tokenize=True))
    assert length==len(rendered)==len('Be concise.|Question')


def test_invalid_instruction_is_rejected():
    with pytest.raises(ValueError,match='must be text'):
        rollout_messages([],{'rollout_system_prompt':123})
