import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib  
# =========================================================
# 1. 환경 설정
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

CFG = {
    'BATCH_SIZE': 256,
}

print("데이터 로드 시작")
# test = pd.read_csv("../test_.csv")
test = pd.read_parquet("./data/test.parquet", engine="pyarrow") #full

# 범주형 결측치 처리
test['gender'].fillna(2, inplace=True)
test['age_group'].fillna(1, inplace=True)
print("데이터 로드 완료")

# =========================================================
# 3. Feature 구성
# =========================================================
target_col = "clicked"
FEATURE_EXCLUDE = {target_col, "seq", "ID"}
feature_cols = [c for c in test.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
history_cols = [c for c in feature_cols if c.startswith("history_")]

print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)} | History features: {len(history_cols)}")

# =========================================================
# 4. 범주형 인코딩 (학습 때 쓰인 encoder 재사용)
# =========================================================
from sklearn.preprocessing import LabelEncoder
import joblib
cat_encoders = joblib.load("./models_weight/hdcn_noseq_cat_encoders.pkl")

for col in cat_cols:
    le = cat_encoders[col]
    test[col] = le.transform(test[col].astype(str).fillna("UNK"))

onehot_cols = [c for c in cat_cols if len(cat_encoders[c].classes_) < 10]
emb_cols = [c for c in cat_cols if c not in onehot_cols]
print(f"One-hot cols: {onehot_cols} | Embedding cols: {emb_cols}")

# =========================================================
# 5. Dataset & Collate 정의 (학습 코드와 동일해야 함)
# =========================================================
class ClickDataset(torch.utils.data.Dataset):
    def __init__(self, df, num_cols, emb_cols, onehot_cols, history_cols):
        self.num_X = df[num_cols].astype(float).fillna(0).values
        self.emb_X = df[emb_cols].astype(int).values if len(emb_cols) > 0 else np.zeros((len(df), 0))
        self.onehot_X = pd.get_dummies(df[onehot_cols].astype(str), columns=onehot_cols).values if len(onehot_cols) > 0 else np.zeros((len(df), 0))
        self.history_X = df[history_cols].astype(float).fillna(0).values

    def __len__(self):
        return len(self.num_X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.num_X[idx], dtype=torch.float),
            torch.tensor(self.emb_X[idx], dtype=torch.long),
            torch.tensor(self.onehot_X[idx], dtype=torch.float),
            torch.tensor(self.history_X[idx], dtype=torch.float)
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
# 6. 모델 클래스 (학습 때 쓰인 구조 그대로)
# =========================================================
import torch.nn as nn

class CorrFeatureAttention(nn.Module):
    def __init__(self, num_features, init_weights=None):
        super().__init__()
        self.attn = nn.Parameter(torch.ones(num_features))
        if init_weights is not None:
            with torch.no_grad():
                self.attn.copy_(torch.tensor(init_weights, dtype=torch.float))
    
    def forward(self, x):
        weights = torch.softmax(self.attn, dim=0)
        return (x * weights).sum(dim=1, keepdim=True)

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x

class WideDeepCTR(nn.Module):
    def __init__(self, num_features, emb_cardinalities, onehot_dim,
                 history_dim, init_history_weights=None,
                 emb_dim=16, hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in emb_cardinalities
        ])
        cat_emb_dim = emb_dim * len(emb_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.history_attn = CorrFeatureAttention(history_dim, init_history_weights)
        input_dim = num_features + cat_emb_dim + onehot_dim + 1
        self.cross = CrossNetwork(input_dim, num_layers=2)
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
        h_attn = self.history_attn(history_x)
        z = torch.cat([num_x, emb_cat, onehot_x.to(num_x.device), h_attn], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)

# =========================================================
# 7. 모델 로드
# =========================================================
print("모델 로드 중...")
ckpt = torch.load("./models_weight/hdcn_noseq.pth", map_location=device)  # 학습 시 저장한 state_dict 경로
emb_cardinalities = [len(cat_encoders[c].classes_) for c in emb_cols]
onehot_dim = pd.get_dummies(test[onehot_cols].astype(str), columns=onehot_cols).shape[1] if len(onehot_cols)>0 else 0
init_weights = np.ones(len(history_cols)) / len(history_cols)  # attention 초기화 대체값

model = WideDeepCTR(
    num_features=len(num_cols),
    emb_cardinalities=emb_cardinalities,
    onehot_dim=onehot_dim,
    history_dim=len(history_cols),
    init_history_weights=init_weights
).to(device)

model.load_state_dict(ckpt)
model.eval()
print("✅ 모델 로드 완료")

# =========================================================
# 8. 추론
# =========================================================
test_dataset = ClickDataset(test, num_cols, emb_cols, onehot_cols, history_cols)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_infer, pin_memory=True)

print("추론 시작")
outs = []
with torch.no_grad():
    for num_x, emb_x, onehot_x, history_x in tqdm(test_loader, desc="[Inference]"):
        num_x, emb_x, onehot_x, history_x = num_x.to(device), emb_x.to(device), onehot_x.to(device), history_x.to(device)
        outs.append(torch.sigmoid(model(num_x, emb_x, onehot_x, history_x)).cpu())

test_preds = torch.cat(outs).numpy()
print("추론 완료")

# =========================================================
# 9. 제출 파일 저장
# =========================================================
submit = pd.read_csv('./data/sample_submission.csv')
submit['clicked'] = test_preds
submit.to_csv('./output/hdcn.csv', index=False)
print("제출 파일 저장 완료: hdcn.csv")
