import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from Model_MHA import MultiHeadAttentionLayer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")




class AttentionLayerDrug2Protein(nn.Module):
    """
    Two Query:
      1) drug2 => Q，K/V = drug1
      2) protein => Q，K/V = drug1
    """
    def __init__(self, hid_dim=256, heads=1, dropout=0.2, device=DEVICE):
        super().__init__()
        self.attention_drug2 = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=heads,
            dropout=dropout,
            device=device
        )
        self.attention_protein = MultiHeadAttentionLayer(
            hid_dim=hid_dim,
            n_heads=heads,
            dropout=dropout,
            device=device
        )

        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, drug1, drug2, protein):
        
        drug1_3d = drug1.unsqueeze(1) if drug1.dim() == 2 else drug1
        drug2_3d = drug2.unsqueeze(1) if drug2.dim() == 2 else drug2
        protein_3d = protein.unsqueeze(1) if protein.dim() == 2 else protein

        # 1) drug2 => Query, drug1 => Key/Value
        att_drug2, _ = self.attention_drug2(drug2_3d, drug1_3d, drug1_3d)
        # 2) protein => Query, drug1 => Key/Value
        att_protein, _ = self.attention_protein(protein_3d, drug1_3d, drug1_3d)


        att_drug2 = att_drug2 + drug2_3d
        att_protein = att_protein + protein_3d


        x_att = att_drug2 + att_protein  # [B, 1, d]
        x_att = self.ln(x_att)
        return x_att.squeeze(1)         # => [B, d]


class DenseLayers(nn.Module):


    def __init__(self, fc_layer_num=2, fc_layer_dim=[256, 128], dropout_rate=0.2):
        super().__init__()

        self.fc_drug1_reduce = nn.Linear(768, 256)
        self.fc_drug2_reduce = nn.Linear(768, 256)
        self.fc_prot_reduce  = nn.Linear(768, 256)

        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()   # BatchNorm
        self.dropout_layers = nn.ModuleList()

        input_dim = 256 * 4  # x_att + drug1_256 + drug2_256 + protein_256
        for out_dim in fc_layer_dim:
            self.fc_layers.append(nn.Linear(input_dim, out_dim))
            self.bn_layers.append(nn.BatchNorm1d(out_dim))
            self.dropout_layers.append(nn.Dropout(p=dropout_rate))
            input_dim = out_dim

        self.fc_output = nn.Linear(fc_layer_dim[-1], 1)

    def forward(self, x_att, drug1, drug2, protein):
        """
        x_att:   [B, 256] 
        drug1:   [B, 768]
        drug2:   [B, 768]
        protein: [B, 768]
        """
     
        d1_256 = F.relu(self.fc_drug1_reduce(drug1))
        d2_256 = F.relu(self.fc_drug2_reduce(drug2))
        p_256  = F.relu(self.fc_prot_reduce(protein))

       
        fusion = torch.cat([d1_256, d2_256, p_256,x_att], dim=1)  # => [B, 1024]

        # MLP: Linear -> BatchNorm -> ReLU -> Dropout
        for fc_layer, bn_layer, drop in zip(self.fc_layers, self.bn_layers, self.dropout_layers):
            fusion = fc_layer(fusion)
            fusion = bn_layer(fusion)   # BN
            fusion = F.relu(fusion)
            fusion = drop(fusion)

        out = self.fc_output(fusion)
        return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class LKE_DTA(nn.Module):

    def __init__(self,
                 n_output=1,
                 output_dim=256,
                 dropout=0.2,
                 heads=1,
                 fc_layer_num=2,
                 fc_layer_dim=[256,128]):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        
        # 1.1 Drug1 (x): [400 -> 768]
        self.fc_drug1 = nn.Linear(400, 768)

        # 1.2 Drug2 (feature): [1536 -> 1024 -> 768]
        self.fc_d2_1 = nn.Linear(1536, 1024)
        self.bn_d2_1 = nn.BatchNorm1d(1024)  # BatchNorm for Drug2
        self.fc_d2_2 = nn.Linear(1024, 768)

        # 1.3 Protein (xt): [2560 -> 1280 -> 768]
        self.fc_p_1 = nn.Linear(2560, 1280)
        self.bn_p_1 = nn.BatchNorm1d(1280)  # BatchNorm for Protein
        self.fc_p_2 = nn.Linear(1280, 768)

       
        self.fc_drug1_256   = nn.Linear(768, 256)
        self.bn_drug1_256   = nn.BatchNorm1d(256)
        self.fc_drug2_256   = nn.Linear(768, 256)
        self.bn_drug2_256   = nn.BatchNorm1d(256)
        self.fc_protein_256 = nn.Linear(768, 256)
        self.bn_protein_256 = nn.BatchNorm1d(256)

       
        self.attention_layer = AttentionLayerDrug2Protein(
            hid_dim=256,
            heads=heads,
            dropout=dropout,
            device=DEVICE
        )

       
        self.dense_layers = DenseLayers(
            fc_layer_num=fc_layer_num,
            fc_layer_dim=fc_layer_dim,
            dropout_rate=dropout
        )
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, data):
        # =========== 1) Drug1 embedding ===========
        x = data.x.view(-1, 400)             # [B, 400]
        drug1 = self.relu(self.fc_drug1(x))  # => [B, 768]

        # =========== 2) Drug2 embedding ===========
        d2 = data.feature.view(-1, 1536)             # [B, 1536]
        d2 = self.relu(self.bn_d2_1(self.fc_d2_1(d2)))  # BatchNorm
        d2 = F.dropout(d2, p=0.2, training=self.training)
        d2 = self.relu(self.fc_d2_2(d2))             # => [B, 768]

        # =========== 3) Protein embedding ===========
        p = data.target.view(-1, 2560)               # [B, 2560]
        p = self.relu(self.bn_p_1(self.fc_p_1(p)))   # BatchNorm
        p = self.dropout(p)
        p = self.relu(self.fc_p_2(p))                # => [B, 768]

        
        drug1_256   = self.bn_drug1_256(self.relu(self.fc_drug1_256(drug1)))
        drug2_256   = self.bn_drug2_256(self.relu(self.fc_drug2_256(d2)))
        protein_256 = self.bn_protein_256(self.relu(self.fc_protein_256(p)))

       
        x_att = self.attention_layer(
            drug1=drug1_256,
            drug2=drug2_256,
            protein=protein_256
        )  # => [B, 256]
        x_att = self.layer_norm(x_att)
        out = self.dense_layers(x_att, drug1, d2, p)  # => [B, 1]
        return out
