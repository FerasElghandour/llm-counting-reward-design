"""Deterministic character-count rewards for structured language-model outputs."""
import re

def target_count(word,letter):
    if not re.fullmatch('[a-z]+',word.lower()) or not re.fullmatch('[a-z]',letter.lower()):raise ValueError('Alphabetic word and one target letter required')
    return word.lower().count(letter.lower())

def parse_answer(response):
    matches=re.findall(r'<answer>\s*(\d+)\s*</answer>',response)
    return int(matches[0]) if len(matches)==1 else None

def score(response,word,letter):
    expected=target_count(word,letter);answer=parse_answer(response)
    formatted=bool(re.fullmatch(r'\s*<reasoning>.*?</reasoning>\s*<answer>\s*\d+\s*</answer>\s*',response,re.DOTALL))
    return {'expected':expected,'parsed_answer':answer,'correctness_reward':2. if answer==expected else -1.,'format_reward':1. if formatted else -1.}

def batch_rewards(completions,words,letters):
    if not len(completions)==len(words)==len(letters):raise ValueError('Aligned completion and target lists required')
    values=[]
    for completion,word,letter in zip(completions,words,letters):
        response=completion[0]['content'];record=score(response,word,letter)
        values.append(record['correctness_reward']+record['format_reward'])
    return values
