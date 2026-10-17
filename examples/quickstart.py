"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from rewards import score,batch_rewards
response='<reasoning>Three matching characters.</reasoning><answer>3</answer>'
print(score(response,'banana','a'))
print(batch_rewards([[{'content':response}]],['banana'],['a']))
