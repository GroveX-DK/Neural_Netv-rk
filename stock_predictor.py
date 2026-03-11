"""
Neural netværk til aktieprisforudsigelse med PyTorch.
Træner på data indtil 2024, evaluerer på data fra 2024 (facit).
"""

import copy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# ============ KONFIGURATION – juster her ============
SEQUENCE_LENGTH = 60       # Antal dage historik til hver forudsigelse
BATCH_SIZE = 16            # Mindre = langsommere træning, ofte bedre (fx 16)
EPOCHS = 200               # Flere epoker = længere træning (fx 300-500)
LEARNING_RATE = 0.0005     # Lavere = langsommere læring (fx 0.0001-0.001)
VALIDATION_SPLIT = 0.15    # Sidste 15% af træningsdata til validering
EARLY_STOPPING_PATIENCE = 100  # Stop hvis ingen forbedring i X epoker (øg ved flere EPOCHS)
# ====================================================


def load_and_prepare_data(csv_path: str):
    """Læs CSV og split ved 2024."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train_df = df[df["date"] < "2024-01-01"].copy()
    test_df = df[df["date"] >= "2024-01-01"].copy()

    return train_df, test_df


def create_sequences(data: np.ndarray, seq_length: int, target_idx: int):
    """Opret sliding window sekvenser med target."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class StockDataset(Dataset):
    """PyTorch Dataset til aktiedata."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM-model til aktieprisforudsigelse."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def train_model(model, train_loader, val_loader, device):
    """Træn modellen med early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping ved epoch {epoch + 1}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    return model


def evaluate_model(model, test_loader, device):
    """Evaluer modellen på testdata (facit)."""
    model.eval()
    predictions = []
    actuals = []
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
            total_loss += criterion(pred, y_batch.to(device)).item()

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

    return predictions, actuals, {"MSE": mse, "MAE": mae, "MAPE": mape}


def save_results_md(
    actuals: np.ndarray,
    predictions: np.ndarray,
    test_dates,
    metrics: dict,
    timestamp: str,
    save_path: Path,
    plot_filename: str = "",
):
    """Gem resultater til en Markdown-fil."""
    # Slutpris (sidste dag)
    end_actual = float(actuals[-1])
    end_predicted = float(predictions[-1])
    end_error_pct = abs(end_actual - end_predicted) / end_actual * 100

    # Første dag
    first_actual = float(actuals[0])
    first_predicted = float(predictions[0])
    first_error_pct = abs(first_actual - first_predicted) / first_actual * 100

    # Fejlstatistik
    errors_pct = np.abs(actuals - predictions) / (actuals + 1e-8) * 100
    avg_error_pct = float(np.mean(errors_pct))
    max_error_pct = float(np.max(errors_pct))
    min_error_pct = float(np.min(errors_pct))

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plot_section = f"\n## Visualisering\n\n![Faktisk vs. forudsigelse]({plot_filename})\n\n" if plot_filename else ""
    md = f"""# Træningsresultater – Goldman Sachs aktieprisforudsigelse

**Genereret:** {gen_time}
{plot_section}
## Slutpris (sidste dag i testperioden)

| | Pris |
|---|---|
| **Forudsigelse** | {end_predicted:.2f} |
| **Faktisk** | {end_actual:.2f} |
| **Fejl i %** | {end_error_pct:.2f}% |

## Første dag i testperioden

| | Pris |
|---|---|
| **Forudsigelse** | {first_predicted:.2f} |
| **Faktisk** | {first_actual:.2f} |
| **Fejl i %** | {first_error_pct:.2f}% |

## Metrikker (hele testperioden)

| Metrik | Værdi |
|---|---|
| MSE | {metrics['MSE']:.4f} |
| MAE | {metrics['MAE']:.4f} |
| MAPE | {metrics['MAPE']:.2f}% |
| Gns. fejl i % | {avg_error_pct:.2f}% |
| Maks. fejl i % | {max_error_pct:.2f}% |
| Min. fejl i % | {min_error_pct:.2f}% |

## Træningsindstillinger

- Epoker: {EPOCHS}
- Learning rate: {LEARNING_RATE}
- Batch size: {BATCH_SIZE}
- Sekvenslængde: {SEQUENCE_LENGTH} dage
- Antal forudsigelser: {len(predictions)}
- Testperiode: {str(test_dates[0])[:10]} til {str(test_dates[-1])[:10]}
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(md, encoding="utf-8")
    print(f"Resultater gemt til {save_path}")


def plot_results(actuals, predictions, test_dates, save_path: str):
    """Visualiser faktisk vs. forudsigelse."""
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actuals, label="Faktisk pris", alpha=0.8)
    plt.plot(test_dates, predictions, label="Forudsigelse", alpha=0.8)
    plt.xlabel("Dato")
    plt.ylabel("Close-pris")
    plt.title("Goldman Sachs: Faktisk vs. forudsigelse (fra 2024)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot gemt til {save_path}")


def main():
    csv_path = Path(__file__).parent / "goldmansachs.csv"
    if not csv_path.exists():
        print(f"Fejl: Fandt ikke {csv_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bruger device: {device}")
    print(f"Træningsindstillinger: EPOCHS={EPOCHS}, LR={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}")

    # 1. Load og split
    train_df, test_df = load_and_prepare_data(csv_path)
    print(f"Træningsdata: {len(train_df)} rækker (indtil 2024)")
    print(f"Testdata (facit): {len(test_df)} rækker (fra 2024)")

    # 2. Features og target
    feature_cols = ["open", "high", "low", "close", "volume"]
    target_col = "close"
    target_idx = feature_cols.index(target_col)

    train_values = train_df[feature_cols].values
    test_values = test_df[feature_cols].values

    # 3. Normalisering (fit kun på træning)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    # 4. Kombiner for at lave sekvenser der krydser split (test behøver seq_length historik)
    full_scaled = np.vstack([train_scaled, test_scaled])
    full_dates = pd.concat([train_df["date"], test_df["date"]]).reset_index(drop=True)

    train_end_idx = len(train_df)
    test_start_idx = train_end_idx - SEQUENCE_LENGTH

    # Træningssekvenser (kun fra træningsdata)
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, target_idx)

    # Valideringssplit (sidste 15% af træningsdata)
    val_size = int(len(X_train) * VALIDATION_SPLIT)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Testsekvenser (bruger data fra train_end - seq_length til slut for at få sekvenser)
    test_data_for_seq = full_scaled[test_start_idx:]
    X_test, y_test = create_sequences(test_data_for_seq, SEQUENCE_LENGTH, target_idx)
    test_dates = full_dates.iloc[test_start_idx + SEQUENCE_LENGTH :].values

    # 5. Datasets og DataLoaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 6. Model
    model = LSTMModel(input_size=len(feature_cols), hidden_size=128, num_layers=2, dropout=0.2).to(device)

    # 7. Træning
    print("\nStarter træning...")
    train_model(model, train_loader, val_loader, device)

    # 8. Evaluer på testdata (facit)
    print("\nEvaluerer på testdata (facit)...")
    predictions, actuals, metrics = evaluate_model(model, test_loader, device)

    print("\n--- Resultater (facit = data fra 2024) ---")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    # 9. Transformér tilbage til original skala for plot (inverse MinMax)
    close_min = scaler.data_min_[target_idx]
    close_max = scaler.data_max_[target_idx]
    actuals_orig = actuals * (close_max - close_min) + close_min
    predictions_orig = predictions * (close_max - close_min) + close_min

    # 10. Plot og resultat-rapport
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    plot_dir = Path(__file__).parent / "Prediction_pictures"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f"prediction_plot_{timestamp}.png"
    plot_results(actuals_orig, predictions_orig, test_dates, str(plot_path))

    results_dir = Path(__file__).parent / "Prediction_pictures"
    results_path = results_dir / f"resultater_{timestamp}.md"
    save_results_md(actuals_orig, predictions_orig, test_dates, metrics, timestamp, results_path, plot_path.name)

    # Gem model
    model_path = Path(__file__).parent / "stock_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_min": scaler.data_min_,
            "scaler_max": scaler.data_max_,
            "feature_cols": feature_cols,
            "seq_length": SEQUENCE_LENGTH,
        },
        model_path,
    )
    print(f"\nModel gemt til {model_path}")


if __name__ == "__main__":
    main()
