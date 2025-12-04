import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        # x: (B, N, L)
        A_V = self.attention_V(x)               # (B, N, D)
        A_U = self.attention_U(x)               # (B, N, D)
        A = self.attention_weights(A_V * A_U)   # (B, N, K)
        A = torch.transpose(A, 1, 2)            # (B, K, N)
        return A                                # (B, K, N)

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class ABMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(ABMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x, mask=None): 
        # Original X: (B, N, L)
        #  - B: batch size
        #  - N: number of instances in the bag
        #  - L: feature dimension of each instance
        # Mask: (B, N)
        #  - 1: Valid
        #  - 0: Padded
        
        med_feat = self.dimreduction(x)             # (B, N, D_inner)
        A = self.attention(med_feat)                # (B, K=1, N)
        
        # Mask padded instances (for Softmax: numerically masking is by setting to massively negative number)
        if mask is not None:
            mask = mask.unsqueeze(1)                # (B, 1, N)
            A = A.masked_fill(mask == 0, -1e9)      # (B, K=1, N)

        # Perform Softmax
        A = F.softmax(A, dim=2)                     # softmax over N
       
        # Batch Matrix Multiplication (apply attention weight to instance feature)
        afeat = torch.bmm(A, med_feat)              # (B, K=1, N) x (B, N, D_inner) = (B, K=1, D_inner) (mult along idx 1 and 2)
        
        # Remove K-dimension
        afeat = afeat.squeeze(1)                    # (B, D_inner)
        
        # Classification    
        outputs = self.classifier(afeat)
        return outputs, med_feat, afeat             # Med_feat: instance lvl embedding (N x D_inner);  afeat: bag lvl embedding (1 x D_inner)
    

class ContrastiveLossManual(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
            
    def forward(self, instance_embedding, bag_embedding, mask, device):
        # # Instance Embeddings : (B, N, D_inner) whereby N is max across bag
        # # Bag Embedding:      : (B, D_inner)
        # # Mask:               : (B, N)

        # # Accumulator
        loss_partial = torch.tensor(0.0, device=instance_embedding.device)
        
        # 1st loop: over bags in batch
        batch_size = instance_embedding.shape[0]

        for B in range(batch_size):
            # 2nd loop: over instances in bag
            num_instances_in_bag = mask[B].sum().item()

            for N in range(int(np.rint(num_instances_in_bag))):
                # Compute numerator
                numerator = torch.exp(
                    F.cosine_similarity(
                        instance_embedding[B, N, :],
                        bag_embedding[B, :],
                        dim=0
                    )
                    / self.temperature
                )

                # Compute Denominator
                denominator = 0.0

                # 3rd loop: Over all bags for contrastive pairs 
                for k in range(batch_size):
                    if k == B:
                        continue
                    denominator += torch.exp(
                        F.cosine_similarity(
                            instance_embedding[B, N, :],
                            bag_embedding[k, :],
                            dim=0
                        )
                        / self.temperature
                    )
                
                # Logarithm
                loss_partial += (torch.log(numerator) - torch.log(denominator))
        
        # Count total number of instances in entire batch
        total_instances = mask.sum().item()
        loss = (-1.0 * loss_partial) / total_instances
        return loss


class ContrastiveLossVectorized(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, instance_emb, bag_emb, mask, device):
        B, N, D = instance_emb.shape
        
        inst = instance_emb.reshape(B*N, D)
        bags = bag_emb
        flat_mask = mask.reshape(B*N)

        valid_idx = flat_mask.nonzero(as_tuple=True)[0]       # (M,)
        inst = inst[valid_idx]                                # (M, D)
        bag_indices = (valid_idx // N).to(device)             # (M,)

        # Normalize
        inst = F.normalize(inst, dim=1)
        bags = F.normalize(bags, dim=1)

        # (M × B)
        sim = torch.einsum("md,bd->mb", inst, bags)

        # Positives
        pos = sim[torch.arange(len(valid_idx), device=device), bag_indices]

        numerator = torch.exp(pos / self.temperature)

        # Exclude positive bag in denominator 
        all_exp = torch.exp(sim / self.temperature)           # (M, B)
        denom = all_exp.sum(dim=1) - torch.exp(pos / self.temperature)
        denom = torch.clamp(denom, min=1e-8)
        loss = -torch.log(numerator / (denom + 1e-8))
        return loss.mean()
