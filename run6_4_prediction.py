import re
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, accuracy_score

from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, RobertaModel, BertModel
import warnings
import torch.nn.functional as F

from rdkit import RDLogger
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import rdMolDescriptors
import tempfile, os
import shutil
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xgboost

# 假设 df 是你的数据框

# 首先，将数据框划分为训练集和临时集（包括测试集和验证集）

# 屏蔽所有RDKit的警告
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore")

from tdc.single_pred import Tox
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#df = pd.concat([df1, df2, df3])

# Display some tokenized sequences and their corresponding labels
#seed = 2023
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
#setup_seed(2023)
# df = pd.read_csv("drug_data2/data1.csv")
test_df = pd.read_csv("test_data.csv")


smiles_data_test = test_df['smiles']
labels_data_test = test_df['label']

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # cuda 2

lr_cl = 0.0001
lr = 0.00001
epochs = 100
max_length = 200
basemodel = "seyonec/PubChem10M_SMILES_BPE_450k"
tokenizer = AutoTokenizer.from_pretrained(basemodel)
aug = True
class Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles, labels, aug=True):
        #  self.dataset = pd.read_csv(file)
        self.smiles = smiles
        self.labels = labels

        data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=smiles,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=max_length,
                                           return_tensors='pt',
                                           return_length=True)
        self.aug = aug
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = torch.LongTensor(labels)

    def tokenizer_smiles(self, smiles):  # for aug
        data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=smiles,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=max_length,
                                           return_tensors='pt',
                                           return_length=True)

        return data['input_ids'], data['attention_mask']

    def mask_input_ids(self, input_ids, mask_value, mask_len=0.15, prob_threshold=0.5):
        # batch_size, seq_len = input_ids.shape
        # # 对于每个样本都生成一个概率
        prob = torch.rand(1)[0]
        # for i in range(batch_size):

        # 找到当前样本中第一个0的位置
        # print(input_ids[i])
        zero_index = (input_ids == 2).nonzero(as_tuple=True)[0][0] if 2 in input_ids else len(input_ids)
        # 计算需要mask的数量
        mask_count = int((zero_index - 2) * mask_len)
        # 随机选择需要mask的位置
        mask_indices = torch.randperm(zero_index - 2)[:mask_count] + 1
        # 进行mask
        # if 0 in mask_indices,then delete

        input_ids[mask_indices] = mask_value
        # if mask_count < 1:
        return input_ids

    def shuffle_input_ids(self, input_ids, prob_threshold=1, shuffle_len=0.25):
        # 对于每个样本都生成一个概率
        prob = torch.rand(1)[0]

        # 找到当前样本中第一个2的位置
        end_index = (input_ids == 2).nonzero(as_tuple=True)[0][0] if 2 in input_ids else len(input_ids)
        # 计算需要shuffle的数量
        shuffle_count = end_index - 2
        # 如果需要shuffle的数量小于1，则不进行shuffle

        subtensor = input_ids[1:end_index].clone()
        indices = torch.randperm(end_index - 1)
        subtensor = subtensor[indices]
        input_ids[1: end_index] = subtensor
        # if shuffle_count < 1:
        #     continue
        # # 创建一个不重复的随机索引序列，从索引1开始

        return input_ids

    def randomize_molecule(self, mol):
        mol = Chem.MolFromSmiles(mol)
        atom_indices = list(range(mol.GetNumAtoms()))
        random.shuffle(atom_indices)
        randomized_mol = Chem.RenumberAtoms(mol, atom_indices)
        randomized_mol = Chem.MolToSmiles(randomized_mol, canonical=False)
        return randomized_mol

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        #  smile = self.smiles[i]

        input_ids = self.input_ids[i]
        label = self.labels[i]
        attention_mask = self.attention_mask[i]
        if self.aug== "Train":
            if torch.rand(1)[0] <= 0.35:
                smiles1 = self.randomize_molecule(self.smiles[i])
            else:
                smiles1 = self.smiles[i]
            input_ids, attention_mask = self.tokenizer_smiles([smiles1])
            input_ids = self.mask_input_ids(input_ids[0], mask_value=4, mask_len=0.15, prob_threshold=0.3)
            return input_ids, attention_mask[0], label

        if not self.aug:

            return input_ids, attention_mask, label

        smiles1 = self.randomize_molecule(self.smiles[i])
        smiles2 = self.randomize_molecule(self.smiles[i])
        input_ids, attention_mask = self.tokenizer_smiles([smiles1, smiles2])
        input_ids1 = self.mask_input_ids(input_ids[0], mask_value=4, mask_len=0.15, prob_threshold=0.5)
        # input_ids1 = self.shuffle_input_ids(input_ids1, prob_threshold=1, shuffle_len=0.25)

        input_ids2 = self.mask_input_ids(input_ids[1], mask_value=4, mask_len=0.15, prob_threshold=0.5)
        # input_ids2 = self.shuffle_input_ids(input_ids2, prob_threshold=1, shuffle_len=0.25)

        return input_ids1, input_ids2, attention_mask[0], attention_mask[1], label


nbits = 2048
def loss_function(a, b):
    tau = 0.05
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)


classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(768, 100)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(100, 2)),
    # ('added_relu2', nn.ReLU(inplace=True)),
    # ('fc3', nn.Linear(25, 2))
]))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        basemodel = "seyonec/PubChem10M_SMILES_BPE_450k"
        self.chemberta = RobertaModel.from_pretrained(basemodel)
        self.chemberta.init_weights()
        self.relu = torch.nn.ReLU()
        self.cls = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, 100)),

        ]))

        self.cls2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, 100)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(100, 2)),
        ]))

    def forward(self, input_ids, attention_mask):
        out = self.chemberta(input_ids=input_ids,
                             attention_mask=attention_mask,
                             )

        rep = out.pooler_output

        out = self.cls(rep)
        return out,rep


# train_dataset_cl = Dataset(smiles=smiles_data.tolist(), labels=labels_data.tolist(), aug=True)
# test_dataset_cl = Dataset(smiles=smiles_data_test.tolist(), labels=labels_data_test.tolist(), aug=True)
# val_dataset_cl = Dataset(smiles=smiles_data_val.tolist(), labels=labels_data_val.tolist(), aug=True)
# train_loader_cl = DataLoader(train_dataset_cl, batch_size=128, shuffle=True, num_workers=16)
#test_loader_cl = DataLoader(test_dataset_cl, batch_size=1, shuffle=False, num_workers=1)
# val_loader_cl = DataLoader(val_dataset_cl, batch_size=64, shuffle=False, num_workers=16)
# # train_loader_cl =
#
# train_dataset = Dataset(smiles=smiles_data.tolist(), labels=labels_data.tolist(), aug="Train")
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)

test_dataset = Dataset(smiles=smiles_data_test.tolist(), labels=labels_data_test.tolist(), aug=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# val_dataset = Dataset(smiles=smiles_data_val.tolist(), labels=labels_data_val.tolist(), aug=False)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
model = Model().to(device)
model.cls = classifier.to(device)
model.load_state_dict(torch.load("best_model/model_65.pt"))

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_cl)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

EPOCHS = 80
for epoch in range(0):
    losses = []
    val_losses = []
    test_preds = []
    test_labels = []
    test_accuracys = []
    for i, (seq1, seq2, attn_mask1, attn_mask2, label) in enumerate(tqdm(train_loader_cl)):
        model.train()
        seq1 = seq1.to(device)
        seq2 = seq2.to(device)
        attn_mask1 = attn_mask1.to(device)
        attn_mask2 = attn_mask2.to(device)
        label = label.to(device)

        out1 = model(seq1, attn_mask1)
        out2 = model(seq2, attn_mask2)
        loss = loss_function(out1, out2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    # print(f"Epoch {epoch} loss", np.mean(losses))

    with torch.no_grad():
        model.eval()

        for i, (seq1, seq2, attn_mask1, attn_mask2, label) in enumerate(tqdm(val_loader_cl)):
            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            attn_mask1 = attn_mask1.to(device)
            attn_mask2 = attn_mask2.to(device)
            label = label.to(device)

            out1 = model(seq1, attn_mask1)
            out2 = model(seq2, attn_mask2)
            loss = loss_function(out1, out2)
            val_losses.append(loss.item())
    print(f"Epoch {epoch} train loss {np.mean(losses)}, val loss,{np.mean(val_losses)}")

criterier = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

print("Trainging")
model.cls = classifier.to(device)
rep_list = []
for epoch in range(1):

    with torch.no_grad():
        model.eval()
        for i, (seq, attn_mask, label) in enumerate(tqdm(test_loader)):
            seq = seq.to(device)
            attn_mask = attn_mask.to(device)
            label = label.to(device)

            pred,rep = model(seq, attn_mask)
            predicted_labels = torch.argmax(pred, dim=1)

            rep_list.append(rep.cpu().numpy())

###save rep_list
rep_list = np.concatenate(rep_list, axis=0)
np.save("rep_list.npy",rep_list)








