import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Train
def main(args):
    df = pd.read_csv(args.data)

    X = df.select_dtypes(include=[np.number]).drop(columns=["suspicious"], errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=136)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = Autoencoder(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for (x,) in train_loader:
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x,) in val_loader:
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.6f} Val: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.model_out)
            np.save(args.scaler_out, scaler.scale_)
            np.save(args.min_out, scaler.min_)

    print("Training complete")


#    parser = argparse.ArgumentParser()
#    parser.add_argument("data", default="behavioral_features.csv")
#    parser.add_argument("epochs", type=int, default=50)
#    parser.add_argument("batch_size", type=int, default=128)
#    parser.add_argument("lr", type=float, default=1e-3)
#    parser.add_argument("model_out", default="autoencoder.pt")
#    parser.add_argument("scaler_out", default="scaler_scale.npy")
#    parser.add_argument("min_out", default="scaler_min.npy")
#    args = parser.parse_args()
