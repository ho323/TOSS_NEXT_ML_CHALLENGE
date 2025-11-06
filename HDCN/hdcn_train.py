import pandas as pd
import numpy as np
import os, random
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CFG = {
    'BATCH_SIZE': 256,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3,
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("데이터 로드 시작")
# train = pd.read_csv("../train.csv")
train = pd.read_parquet("./data/train.parquet", engine="pyarrow") #full
# test = pd.read_csv("../test_.csv")
test = pd.read_parquet("./data/test.parquet", engine="pyarrow") #full

train['gender'].fillna(2, inplace=True)
test['gender'].fillna(2, inplace=True)

train['age_group'].fillna(1, inplace=True)
test['age_group'].fillna(1, inplace=True)


import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 데이터 전처리
# =========================================================
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("데이터 로드 완료")

target_col = "clicked"
FEATURE_EXCLUDE = {target_col, "seq", "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
history_cols = [c for c in feature_cols if c.startswith("history_")]

print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)} | History features: {len(history_cols)}")

# ---------------------------------------------------------
# 범주형 인코딩
# ---------------------------------------------------------
def encode_categoricals(train_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col]  = le.transform(test_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        print(f"{col} unique categories: {len(le.classes_)}")
    return train_df, test_df, encoders

train, test, cat_encoders = encode_categoricals(train, test, cat_cols)

# 범주 개수 작은 건 one-hot, 나머지는 embedding
onehot_cols = [c for c in cat_cols if len(cat_encoders[c].classes_) < 10]
emb_cols = [c for c in cat_cols if c not in onehot_cols]
print(f"One-hot cols: {onehot_cols} | Embedding cols: {emb_cols}")

# =========================================================
# Dataset & Collate
# =========================================================
class ClickDataset(Dataset):
    def __init__(self, df, num_cols, emb_cols, onehot_cols, history_cols, target_col=None, has_target=True):
        self.num_X = df[num_cols].astype(float).fillna(0).values
        self.emb_X = df[emb_cols].astype(int).values if len(emb_cols) > 0 else np.zeros((len(df), 0))
        self.onehot_X = pd.get_dummies(df[onehot_cols].astype(str), columns=onehot_cols).values if len(onehot_cols) > 0 else np.zeros((len(df), 0))
        self.history_X = df[history_cols].astype(float).fillna(0).values
        self.has_target = has_target
        if has_target:
            self.y = df[target_col].astype(np.float32).values

    def __len__(self):
        return len(self.num_X)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        emb_x = torch.tensor(self.emb_X[idx], dtype=torch.long)
        onehot_x = torch.tensor(self.onehot_X[idx], dtype=torch.float)
        history_x = torch.tensor(self.history_X[idx], dtype=torch.float)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, emb_x, onehot_x, history_x, y
        else:
            return num_x, emb_x, onehot_x, history_x

def collate_fn_train(batch):
    num_x, emb_x, onehot_x, history_x, ys = zip(*batch)
    return (
        torch.stack(num_x),
        torch.stack(emb_x),
        torch.stack(onehot_x),
        torch.stack(history_x),
        torch.stack(ys)
    )

def collate_fn_infer(batch):
    num_x, emb_x, onehot_x, history_x = zip(*batch)
    return (
        torch.stack(num_x),
        torch.stack(emb_x),
        torch.stack(onehot_x),
        torch.stack(history_x)
    )

# =========================================================
# Corr Attention Layer
# =========================================================
class CorrFeatureAttention(nn.Module):
    def __init__(self, num_features, init_weights=None):
        super().__init__()
        self.attn = nn.Parameter(torch.ones(num_features))
        if init_weights is not None:
            with torch.no_grad():
                self.attn.copy_(torch.tensor(init_weights, dtype=torch.float))
    
    def forward(self, x):
        weights = torch.softmax(self.attn, dim=0)  # feature-wise attention
        return (x * weights).sum(dim=1, keepdim=True)  # weighted sum

# =========================================================
# CrossNetwork
# =========================================================
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x

# =========================================================
# WideDeepCTR 모델
# =========================================================
class WideDeepCTR(nn.Module):
    def __init__(self, num_features, emb_cardinalities, onehot_dim,
                 history_dim, init_history_weights=None,
                 emb_dim=16,
                 hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]):
        super().__init__()
        # Embedding
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in emb_cardinalities
        ])
        cat_emb_dim = emb_dim * len(emb_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)

        # Corr Attention
        self.history_attn = CorrFeatureAttention(history_dim, init_history_weights)

        input_dim = num_features + cat_emb_dim + onehot_dim + 1  # +1 for weighted history summary
        self.cross = CrossNetwork(input_dim, num_layers=2)

        # MLP
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [
                nn.Linear(input_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout[i % len(dropout)])
            ]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, emb_x, onehot_x, history_x):
        num_x = self.bn_num(num_x)

        if emb_x.size(1) > 0:
            emb_feats = [emb(emb_x[:, i]) for i, emb in enumerate(self.emb_layers)]
            emb_cat = torch.cat(emb_feats, dim=1)
        else:
            emb_cat = torch.zeros((num_x.size(0),0),device=num_x.device)

        # Corr attention summary
        h_attn = self.history_attn(history_x)

        z = torch.cat([num_x, emb_cat, onehot_x.to(num_x.device), h_attn], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)


def train_model(train_df, num_cols, emb_cols, onehot_cols, history_cols, target_col, batch_size, epochs, lr, device):
    train_dataset = ClickDataset(train_df, num_cols, emb_cols, onehot_cols, history_cols, target_col, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train, pin_memory=True)

    emb_cardinalities = [len(cat_encoders[c].classes_) for c in emb_cols]
    onehot_dim = pd.get_dummies(train_df[onehot_cols].astype(str), columns=onehot_cols).shape[1] if len(onehot_cols)>0 else 0

    # corr 기반 초기화
    corr = train_df[history_cols + [target_col]].corr()[target_col].drop(target_col).fillna(0)
    init_weights = corr.abs() / corr.abs().sum()
    print(f"History corr-based weights: {init_weights.sort_values(ascending=False).head(5)}")

    model = WideDeepCTR(
        num_features=len(num_cols),
        emb_cardinalities=emb_cardinalities,
        onehot_dim=onehot_dim,
        history_dim=len(history_cols),
        init_history_weights=init_weights.values,
        emb_dim=16
    ).to(device)

    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    print("학습 시작")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for num_x, emb_x, onehot_x, history_x, ys in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            num_x, emb_x, onehot_x, history_x, ys = num_x.to(device), emb_x.to(device), onehot_x.to(device), history_x.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(num_x, emb_x, onehot_x, history_x)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * ys.size(0)
        total_loss /= len(train_dataset)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print("학습 완료")
    return model

# =========================================================
# 실행
# =========================================================
print("모델 학습 실행")
model = train_model(
    train_df=train,
    num_cols=num_cols,
    emb_cols=emb_cols,
    onehot_cols=onehot_cols,
    history_cols=history_cols,
    target_col=target_col,
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr=CFG['LEARNING_RATE'],
    device=device
)

torch.save(model.state_dict(), "./models_weight/hdcn_noseq.pth")
joblib.dump(cat_encoders, "./models_weight/hdcn_noseq_cat_encoders.pkl")  
print("모델 및 인코더 저장 완료!")
