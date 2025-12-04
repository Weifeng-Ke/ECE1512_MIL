import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ContrastiveLossManual, ContrastiveLossVectorized

# -----------------------------
# Use toy input to check for agreement
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B = 3      # bags
N = 5      # max instances per bag
D = 4      # feature dimension

torch.manual_seed(0)

instance_emb = torch.randn(B, N, D).to(device)
bag_emb = torch.randn(B, D).to(device)

# Random mask with variable #instances per bag
mask = torch.tensor([
    [1,1,1,0,0],   # bag 0 : 3 instances
    [1,1,0,0,0],   # bag 1 : 2 instances
    [1,1,1,1,0]    # bag 2 : 4 instances
], dtype=torch.float32).to(device)


manual_loss_fn = ContrastiveLossManual().to(device)
vector_loss_fn = ContrastiveLossVectorized().to(device)

# -----------------------------
# Compare the results
# -----------------------------
manual_loss = manual_loss_fn(instance_emb, bag_emb, mask, device)
vector_loss = vector_loss_fn(instance_emb, bag_emb, mask, device)

print("Manual loss    :", manual_loss.item())
print("Vectorized loss:", vector_loss.item())
print("Difference     :", abs(manual_loss - vector_loss).item())
