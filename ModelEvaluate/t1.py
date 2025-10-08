
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

reference = {1: ["a cat sitting on a bench", "a cat on the bench"]}
candidate = {1: ["a cat is sitting on the bench"]}

# 初始化CIDEr和SPICE评分器
cider_scorer = Cider()
spice_scorer = Spice()

# 计算CIDEr和SPICE评分
cider_score, _ = cider_scorer.compute_score(reference, candidate)
spice_score, _ = spice_scorer.compute_score(reference, candidate)

print(f"CIDEr Score: {cider_score}")
print(f"SPICE Score: {spice_score}")
