import sys
import torch
from geda_model import GEDAModel
from causal_crisis_model import CausalCrisisModel

# Initialize models
geda = GEDAModel()
causal = CausalCrisisModel()

# Count params
geda_params = sum(p.numel() for p in geda.parameters() if p.requires_grad)
causal_params = sum(p.numel() for p in causal.parameters() if p.requires_grad)

diff = causal_params - geda_params
diff_percent = (diff / geda_params) * 100

print(f"GEDA Model Parameters: {geda_params:,}")
print(f"CausalCrisis Model Parameters: {causal_params:,}")
print(f"Difference: +{diff:,} ({diff_percent:.2f}%)")

# Break down CausalCrisis specific modules
print("\nBreakdown of new modules in CausalCrisis:")
dis_img = sum(p.numel() for p in causal.disentangle_img.parameters() if p.requires_grad)
dis_txt = sum(p.numel() for p in causal.disentangle_txt.parameters() if p.requires_grad)
dom_cv = sum(p.numel() for p in causal.domain_cls_cv.parameters() if p.requires_grad)
dom_ct = sum(p.numel() for p in causal.domain_cls_ct.parameters() if p.requires_grad)
dom_sv = sum(p.numel() for p in causal.domain_cls_sv.parameters() if p.requires_grad)
dom_st = sum(p.numel() for p in causal.domain_cls_st.parameters() if p.requires_grad)

print(f"Disentangler IMG: {dis_img:,}")
print(f"Disentangler TXT: {dis_txt:,}")
print(f"Domain Cls (C_V, C_T, S_V, S_T): {dom_cv*4:,}")
