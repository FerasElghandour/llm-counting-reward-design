from rewards import target_count,score,batch_rewards
response='<reasoning>Three matching characters.</reasoning><answer>3</answer>'
assert target_count('banana','a')==3
assert score(response,'banana','a')['correctness_reward']==2
assert score(response,'banana','a')['format_reward']==1
assert score('<answer>3</answer><answer>3</answer>','banana','a')['correctness_reward']==-1
assert batch_rewards([[{'content':response}]],['banana'],['a'])==[3.]
try:target_count('','a')
except ValueError:pass
else:raise AssertionError('Empty word accepted')
try:batch_rewards([],['word'],['a'])
except ValueError:pass
else:raise AssertionError('Misaligned targets accepted')
print('Exact targets, structured format, duplicate answers, and completion alignment passed.')
