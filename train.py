import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.lstm_model import InsiderLSTM

data = pd.read_csv("data/behavioral_features.csv")
data['day'] = pd.to_datetime(data['day'])
data = data.sort_values(['user','day'])

data['future_risk'] = 0
for user, df in data.groupby('user'):
    idx = df.index
    risky = df[df['suspicious']==1].index
    for i in risky:
        pos = list(idx).index(i)
        data.loc[idx[max(0,pos-7):pos],'future_risk']=1

FEATURES = ['logins','off_hour_logins','file_ops','files_copied',
            'usb_events','http_uploads','emails_sent']

X, y = [], []
for user, df in data.groupby('user'):
    v = df[FEATURES].values
    l = df['future_risk'].values
    for i in range(len(df)-7):
        X.append(v[i:i+7])
        y.append(l[i+7])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                  torch.tensor(y_train,dtype=torch.float32)),
    batch_size=256, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = InsiderLSTM().to(device)

pos_weight = torch.tensor([(1-y_train.mean())/y_train.mean()]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(5):
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print("Epoch",e+1,"done")

torch.save(model.state_dict(),"model.pt")
