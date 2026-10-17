"""Recovered original GPU training workflow;5+80 updates share one adapter.

This path requires a compatible CUDA/Unsloth/vLLM/TRL environment. It has not
been rerun on the current CPU host and does not establish a new model result.
"""
import torch
if not torch.cuda.is_available():raise SystemExit("CUDA GPU required; current CPU host cannot reproduce the original GRPO workflow")

# Original code cell 5
import unsloth
from unsloth import FastLanguageModel
import torch
max_seq_length = 512
lora_rank = 64
model, tokenizer = FastLanguageModel.from_pretrained(model_name='Qwen/Qwen2.5-3B-Instruct', max_seq_length=max_seq_length, load_in_4bit=True, fast_inference=True, max_lora_rank=lora_rank, gpu_memory_utilization=0.5)
model = FastLanguageModel.get_peft_model(model, r=lora_rank, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_alpha=lora_rank, use_gradient_checkpointing='unsloth')

# Original code cell 7
SYSTEM_PROMPT = ''
USER_PROMPT = 'How many of the letter "g" are there in the word "engage"'
text_for_completion = tokenizer.apply_chat_template(conversation=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': USER_PROMPT}], tokenize=False, add_generation_prompt=True)
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
output = model.fast_generate([text_for_completion], sampling_params=sampling_params, lora_request=None)[0].outputs[0].text
print('=== TEXT FOR COMPLETION ===')
print(text_for_completion)
print('=== GENERATED OUTPUT ===')
print(output)

# Original code cell 9
SYSTEM_PROMPT = 'Count occurrences of the requested letter in the word. Work step by step.\nInside <reasoning>, spell the word once, using numbered lines in the original\norder. On each line show the letter and the running count of the requested\nletter. Increase the count only when that letter matches. Include every letter\nexactly once. Then put only the final integer inside <answer>.\n\nExample question: How many of the letter "o" are there in the word "room"?\nExample response:\n<reasoning>\nLetter-by-letter spelling:\n1. r - 0 so far\n2. o - 1 so far\n3. o - 2 so far\n4. m - 2 so far\n</reasoning>\n<answer>\n2\n</answer>'
USER_PROMPT = 'How many of the letter "g" are there in the word "engage"'
text_for_completion = tokenizer.apply_chat_template(conversation=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': USER_PROMPT}], tokenize=False, add_generation_prompt=True)
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
output = model.fast_generate([text_for_completion], sampling_params=sampling_params, lora_request=None)[0].outputs[0].text
print('=== TEXT FOR COMPLETION ===')
print(text_for_completion)
print('=== GENERATED OUTPUT ===')
print(output)

# Original code cell 12
ALL_WORDS = ['idea', 'glow', 'rust', 'maze', 'echo', 'wisp', 'veto', 'lush', 'gaze', 'knit', 'fume', 'plow', 'void', 'oath', 'grim', 'crisp', 'lunar', 'fable', 'quest', 'verge', 'brawn', 'elude', 'aisle', 'ember', 'crave', 'ivory', 'mirth', 'knack', 'wryly', 'onset', 'mosaic', 'velvet', 'sphinx', 'radius', 'summit', 'banner', 'cipher', 'glisten', 'mantle', 'scarab', 'expose', 'fathom', 'tavern', 'fusion', 'relish', 'lantern', 'enchant', 'torrent', 'capture', 'orchard', 'eclipse', 'frescos', 'triumph', 'absolve', 'gossipy', 'prelude', 'whistle', 'resolve', 'zealous', 'mirage', 'aperture', 'sapphire']
print(len(ALL_WORDS))
ALL_WORDS[:10]

# Original code cell 13
from datasets import Dataset
import random

def generate_records():
    for word in ALL_WORDS:
        for letter in sorted(set(word)):
            yield {'words': word, 'letters': letter, 'counts': word.count(letter)}
        num_letters_not_in_word_left = int(len(word) // 7 + 1)
        random.seed(hash(word))
        all_letters = list('abcdefghijklmnopqrstuvwxyz')
        random.shuffle(all_letters)
        for letter in all_letters:
            if letter not in word:
                yield {'words': word, 'letters': letter, 'counts': 0}
                num_letters_not_in_word_left -= 1
            if num_letters_not_in_word_left == 0:
                break
ds = Dataset.from_generator(generate_records)
ds[0]

# Original code cell 14
import re
from datasets import load_dataset, Dataset
ds = ds.map(lambda x: {'prompt': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': 'How many of the letter "{}" are there in the word "{}"'.format(x['letters'], x['words'])}]})
ds[0]
lengths = [len(tokenizer.apply_chat_template(r['prompt'], tokenize=True, add_generation_prompt=True)) for r in ds]
print('Maximum dataset prompt tokens:', max(lengths))
assert max(lengths) <= 256, 'Few-shot prompt exceeds the prompt budget.'
assert all((r['prompt'][0]['content'] == SYSTEM_PROMPT for r in ds))
print('Every dataset record uses the baseline few-shot SYSTEM_PROMPT.')

# Original code cell 15
text = tokenizer.apply_chat_template(ds[0]['prompt'], tokenize=False, add_generation_prompt=True)
from vllm import SamplingParams
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
output = model.fast_generate([text], sampling_params=sampling_params, lora_request=None)[0].outputs[0].text
print(output)

# Original code cell 18
def extract_letter_numbering(response):
    """Extract the numbers at the beginning of the line

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    4. a - 2 so far
    5. l - 2 so far
    returns [1, 2, 3, 4, 5]
    """
    import re
    pattern = '\\n(\\d+)\\. [a-z]'
    matches = re.findall(pattern, response)
    if matches:
        return [int(m) for m in matches]
    return []
assert extract_letter_numbering('\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. a - 2 so far\n5. l - 2 so far\n') == [1, 2, 3, 4, 5]

def numbering_reward_func(completions, words, **kwargs) -> list[float]:
    """Provides a reward for getting the numbering at the beginning of the line correct

    1. g - 1 so far <-- Good in-order numbering
    2. o - 1 so far <-- Good in-order numbering
    3. a - 2 so far <-- Good in-order numbering
    3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4
    1. l - 2 so far <-- Bad numbering, extra letter and out-of-order
    1. l - 2 so far <-- Bad numbering, extra letter and out-of-order

    """
    responses = [completion[0]['content'] for completion in completions]
    res = []
    for response, word in zip(responses, words):
        reward = 0
        for ix, spell_number in enumerate(extract_letter_numbering(response)):
            line_number = ix + 1
            if spell_number == line_number:
                reward += 0.5
            else:
                reward -= 0.5
            if line_number > len(word):
                reward -= 1.0
        res.append(reward / len(word))
    return res
res = numbering_reward_func(completions=[[{'content': '<reasoning>\nHere is a letter by letter spelling:\n1. g - 1 so far <-- Good in-order numbering\n2. o - 1 so far <-- Good in-order numbering\n3. a - 2 so far <-- Good in-order numbering\n3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4\n1. l - 2 so far <-- Bad numbering, extra letter and out-of-order\n1. l - 2 so far <-- Bad numbering, extra letter and out-of-order\n</reasoning>\n<answer>2</answer>'}], [{'content': '<reasoning>\nHere is a letter by letter spelling:\n1. g - 1 so far <-- Good in-order numbering\n2. o - 1 so far <-- Good in-order numbering\n3. a - 2 so far <-- Good in-order numbering\n3. l - 2 so far <-- Bad numbering, out-of-order, 3 should be 4\n</reasoning>\n<answer>2</answer>'}]], words=['goal', 'goal'])
print(res)
assert res[1] > res[0], 'The better response should have a higher reward'

# Original code cell 20
def extract_spelling(response):
    """Extract the spelling from the response

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    3. l - 2 so far
    5. l - 2 so far
    Returns "goall"
    """
    import re
    pattern = '\\n\\d+. ([a-z])'
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if matches:
        return ''.join([m for m in matches])
    return ''
extract_spelling('Here is a letter by letter spelling:\n\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n3. l - 2 so far\n5. l - 2 so far\n') == 'goall'

def spelling_reward_func(completions, words, **kwargs) -> list[float]:
    """A spelling reward function."""
    from collections import Counter
    responses = [completion[0]['content'] for completion in completions]
    res = []
    for word, response in zip(words, responses):
        reward = 0.0
        proposed = extract_spelling(response)
        reward += 2.0 if proposed == word else 0.0
        reward -= 0.5 * abs(len(proposed) - len(word))
        reward -= 1.0 * sum((Counter(proposed) - Counter(word)).values())
        reward -= 0.5 * sum((Counter(word) - Counter(proposed)).values())
        res.append(reward)
    return res
res = spelling_reward_func(completions=[[{'content': '<reasoning>\nHere is a letter by letter spelling:\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n5. l - 2 so far\n</reasoning>\n<answer>2</answer>'}], [{'content': '<reasoning>\nHere is a letter by letter spelling:\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n</reasoning>\n<answer>2</answer>'}]], words=['goal', 'goal'])
print(res)
assert res[1] > res[0], 'The better response should have a higher reward'

# Original code cell 22
def get_resp_letters_and_counts(response):
    """Extract the letters and counts from the response

    Example:
    1. g - 1 so far
    2. o - 1 so far
    3. a - 2 so far
    4. a - 2 so far
    5. l - 2 so far
    returns [('g', 1), ('o', 1), ('a', 2), ('a', 2), ('l', 2)]
    """
    import re
    pattern = '\\n(\\d+)\\. ([a-z])\\D*(\\d+)'
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if not matches:
        return []
    return [(matched_letter, matched_count_so_far) for _, matched_letter, matched_count_so_far in matches]
assert get_resp_letters_and_counts('\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. a - 2 so far\n5. l - 2 so far\n') == [('g', '1'), ('o', '1'), ('a', '2'), ('a', '2'), ('l', '2')]

def counting_reward_func(completions, letters, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    res = []
    for letter, response in zip(letters, responses):
        reward = 0
        letters_and_counts = get_resp_letters_and_counts(response)
        if not letters_and_counts:
            res.append(-1)
            continue
        actual_count = 0
        for resp_letter, resp_count in letters_and_counts:
            if letter == resp_letter:
                actual_count += 1
            if int(resp_count) == actual_count:
                reward += 1.0
            else:
                reward -= 1.0
        res.append(reward / len(letters_and_counts))
    return res
res = counting_reward_func(completions=[[{'content': '<reasoning>\nHere is a letter by letter spelling:\n\n1. g - 0 so far\n2. o - 0 so far\n3. a - 1 so far\n4. a - 2 so far\n5. l - 0 so far\n\n\n</reasoning>\n<answer>\nThis is my answer.\n</answer>'}], [{'content': '<reasoning>\nHere is a letter by letter spelling:\n\n1. g - 1 so far\n2. o - 1 so far\n3. a - 1 so far\n4. a - 1 so far\n5. l - 1 so far\n\n\n</reasoning>\n<answer>\nThis is my answer.\n</answer>'}]], letters=['g', 'g'])
print(res)
assert res[1] > res[0], 'The better response should have a higher reward'

# Original code cell 24
import re

def extract_xml_answer(text: str) -> str:
    """Extracts the string between <answer> and </answer> tags."""
    import re
    pattern = '<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''
assert extract_xml_answer('\n<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>SUPERCALIFRAGILISTICEXPIALIDOCIOUS</answer>\n') == 'SUPERCALIFRAGILISTICEXPIALIDOCIOUS'

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = '\\s*<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>'
    res = []
    for completion in completions:
        reward = 0.0
        response = completion[0]['content']
        match = re.match(pattern, response, flags=re.MULTILINE | re.DOTALL)
        reward += 0.5 if match else -0.5
        extracted_answer = extract_xml_answer(response)
        reward += 0.5 if extracted_answer.isdigit() else -0.5
        res.append(reward)
    return res
res = format_reward_func(completions=[[{'content': 'This is my answer'}], [{'content': '<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>\n3\n</answer>'}]])
print(res)
assert res[1] > res[0], 'The better response should have a higher reward'

# Original code cell 26
def correct_answer_reward_func(prompts, completions, counts, **kwargs) -> list[float]:
    """Reward the final answer if it is correct."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(f"\n{'-' * 20}\nQuestion: {prompts[0][-1]['content']}\nAnswer: {counts[0]}\nResponse: {responses[0]}\nExtracted: {extracted_responses[0]}\nCorrect: {str(extracted_responses[0]) == str(counts[0])}!\n    ")
    res = [2.0 if str(r) == str(a) else -1.0 for r, a in zip(extracted_responses, counts)]
    return res
res = correct_answer_reward_func(prompts=[[{'content': 'How many...'}], [{'content': 'How many...'}]], completions=[[{'content': '<reasoning>.../reasoning>\n<answer>\n3\n</answer>'}], [{'content': '<reasoning>.../reasoning>\n<answer>\n3\n</answer>'}]], letters=['g', 'g'], counts=[0, 3])
print(res)
assert res[1] > res[0], 'The better response should have a higher reward'

# Original code cell 28
REWARD_FUNCS = [numbering_reward_func, spelling_reward_func, counting_reward_func, format_reward_func, correct_answer_reward_func]

# Original code cell 30
COMMON_GRPO_TRAINING_PARAMS = dict(learning_rate=1e-05, beta=0.0001, per_device_train_batch_size=16, num_generations=4, gradient_accumulation_steps=1, bf16=False, fp16=True, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type='cosine', optim='adamw_8bit', logging_steps=1, max_prompt_length=256, max_completion_length=200, num_train_epochs=1, save_steps=20, save_total_limit=2, max_grad_norm=0.1, report_to='none', output_dir='outputs', use_vllm=True)

# Original code cell 32
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=5)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=REWARD_FUNCS, args=training_args, train_dataset=ds)
trainer_res = trainer.train()

# Original code cell 33
import pandas as pd
import matplotlib.pyplot as plt
print(f'available columns: {trainer.state.log_history[0].keys()}')
log_df = pd.DataFrame(trainer.state.log_history)
log_df['reward'].plot()
log_df['rewards/correct_answer_reward_func/mean'].plot()
plt.legend(['reward', 'rewards/correct_answer_reward_func/mean'])
plt.show()
quick_train_seconds = trainer_res.metrics['train_runtime']
print(f'Five-step quick pass: {quick_train_seconds:.1f} seconds')
print(f'Estimated 80-step run: {quick_train_seconds / 5 * 80 / 60:.1f} minutes')
reward_columns = [c for c in log_df.columns if c.startswith('rewards/') and c.endswith('/mean')]
display(log_df[['step', 'reward'] + reward_columns].dropna(subset=['reward']))
assert log_df['reward'].abs().max() > 0, 'Inspect reward design before the long run.'

# Original code cell 35
training_args = GRPOConfig(**COMMON_GRPO_TRAINING_PARAMS, max_steps=80)
trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=REWARD_FUNCS, args=training_args, train_dataset=ds)
trainer_res = trainer.train()

# Original code cell 36
import pandas as pd
import matplotlib.pyplot as plt
print(f'available columns: {trainer.state.log_history[0].keys()}')
log_df = pd.DataFrame(trainer.state.log_history)
log_df['reward'].plot()
log_df['rewards/correct_answer_reward_func/mean'].plot()
plt.legend(['reward', 'rewards/correct_answer_reward_func/mean'])
plt.show()
log_df['rewards/correct_answer_reward_func/mean'].rolling(10).mean().plot()
plt.title('Correctness reward: 10-step rolling mean')
plt.show()
correctness = log_df['rewards/correct_answer_reward_func/mean'].dropna()
print('First ten correctness mean:', correctness.head(10).mean())
print('Last ten correctness mean:', correctness.tail(10).mean())
display(log_df[['step', 'reward', 'rewards/correct_answer_reward_func/mean']].dropna(subset=['reward']))
from IPython.display import Markdown, display
log_df.to_csv('long_training_metrics.csv', index=False)
correctness = log_df['rewards/correct_answer_reward_func/mean'].dropna()
early = correctness.head(10).mean()
late = correctness.tail(10).mean()
display(Markdown(f'First-ten mean correctness reward: **{early:.5f}**. Last-ten mean: **{late:.5f}**. Change: **{late - early:+.5f}**. These are sampled training rewards, not held-out accuracy. The smoke test and longer run updated the same adapter, giving 85 optimizer steps.'))
if late <= early:
    display(Markdown('The end-window trend was flat or decreasing. Small batches and sampled completions produce noisy rewards. A lower learning rate is tested next as a single controlled change.'))
    params = {**COMMON_GRPO_TRAINING_PARAMS, 'learning_rate': 5e-06, 'output_dir': 'outputs/remedial'}
    remedial = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=REWARD_FUNCS, args=GRPOConfig(**params, max_steps=25), train_dataset=ds)
    remedial.train()
    remedy_df = pd.DataFrame(remedial.state.log_history)
    remedy_df.to_csv('remedial_training_metrics.csv', index=False)
    remedy_df['rewards/correct_answer_reward_func/mean'].plot(title='Remedial run: learning rate 5e-6')
    plt.show()
    avg = remedy_df['rewards/correct_answer_reward_func/mean'].dropna().mean()
    display(Markdown(f'Only learning rate changed, from 1e-5 to 5e-6. Remedial mean correctness reward: **{avg:.5f}**, versus the previous final window **{late:.5f}**. Sampled prompts limit causal interpretation.'))
else:
    display(Markdown('The end-window mean increased. No anomaly-driven remedial run was needed. Remaining reasoning errors must be considered separately from final-answer correctness.'))

# Original code cell 39
model.save_lora('grpo_saved_lora')

# Original code cell 40
def compare_old_and_new_model(messages):
    from vllm import SamplingParams
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
    old = model.fast_generate(text, sampling_params=sampling_params, lora_request=None)[0].outputs[0].text
    new = model.fast_generate(text, sampling_params=sampling_params, lora_request=model.load_lora('grpo_saved_lora'))[0].outputs[0].text
    print('===OLD===\n')
    print(old)
    print('\n\n===NEW===\n')
    print(new)
    return {'old': old, 'new': new}

# Original code cell 42
record = ds[0]
comparison = compare_old_and_new_model(record['prompt'])
print('Expected count:', record['counts'])
for name, response in comparison.items():
    print(name, 'final answer correct:', extract_xml_answer(response) == str(record['counts']))

# Original code cell 45
facts = compare_old_and_new_model([{'role': 'user', 'content': 'What is the capital of the Philippines?'}])
for name, response in facts.items():
    print(name, 'mentions Manila:', 'manila' in response.lower())
print('This single question is a limited retained-knowledge check.')
