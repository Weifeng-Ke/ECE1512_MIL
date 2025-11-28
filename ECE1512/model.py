import torch.nn as nn
import torch
import torch.nn.functional as F


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
        ## x: N x L (instance x feature dimension)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN (head x instances)


        return A  # K x N

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
        # input x[0]: N x D_feat =(N,384)
        x = self.fc1(x)
        # out x: N x m_dim =(N,512)
        x = self.relu1(x)
        # out x: N x m_dim =(N,512)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x    #x : N x m_dim =(N,512)

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
        self.attn_heads = getattr(conf, 'attn_heads', 1)
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, self.attn_heads)
        self.classifier = Classifier_1fc(conf.D_inner *self.attn_heads, conf.n_class, droprate)

    def forward(self, x): ## x: N x L -> (B,N,L) = (1,N,D_feat) Batch,Number of patches/instance per bag, feature dimension
        x = x[0] ## Selecting the frist bag dimension (N,D_feat)=(N,384)
        #input x dimension (N,384)
        med_feat = self.dimreduction(x) ## N x L 
        #output dimension Linear(384 -> 128) + ReLU(+residual )
        #Input to attention med_feat dimension (N,128)
        A = self.attention(med_feat)  ## K x N
        #output A has size (K,N) = (1,N)
        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N for each head
        #Value attention weights A has size (K,N) = (1,N)
        #input A dimension (1,N) and med_feat dimension (N,128)
        afeat = torch.mm(A, med_feat) ## K x L
        #output afeat dimension (K,L) = (1,128)
        #This yields the attended feature vector for the bag

        afeat = afeat.reshape(1, -1) # concatente head-wise pooled embedings

        #Input afeat dimension (1,128)
        outputs = self.classifier(afeat)
        #output dimension Linear(128 -> n_class)-> logits(K,C)=(1,number of classes)
        return outputs
    


    