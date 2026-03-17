"""
Neural netværk til aktieprisforudsigelse med PyTorch.
Træner på de første TRAINING_YEARS år af aktiedata og forudsiger det efterfølgende PREDICTION_YEARS år.
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
STOCK_NAME               = "Goldman Sachs"  # Navn på aktien (bruges i titler/rapporter)
TRAINING_YEARS           = 25               # Antal år at træne på (fra første dato i data)
PREDICTION_YEARS         = 1                # Antal år at forudsige efter træningsperioden
SEQUENCE_LENGTH          = 60               # Antal dage historik til hver forudsigelse
BATCH_SIZE               = 8                # Batch-størrelse under træning
EPOCHS                   = 200              # Maks. epoker pr. træningsrunde
LEARNING_RATE            = 0.0001           # Læringsrate for Adam-optimizeren
VALIDATION_SPLIT         = 0.05             # Andel af træningsdata brugt til validering (5%)
USE_EARLY_STOPPING       = True             # Brug early stopping
EARLY_STOPPING_PATIENCE  = 50               # Epoker uden forbedring før stop
EARLY_STOPPING_MIN_DELTA = 1e-4             # Mindste forbedring i val_loss der tæller (0.0001)
# ====================================================


def load_and_prepare_data(csv_path: str):
    """Læs CSV. Træner på de første TRAINING_YEARS år, forudsiger de næste PREDICTION_YEARS år."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    first_date = df["date"].iloc[0]
    train_end  = first_date + pd.DateOffset(years=TRAINING_YEARS)
    pred_end   = train_end  + pd.DateOffset(years=PREDICTION_YEARS)

    train_df = df[df["date"] < train_end].copy()
    test_df  = df[(df["date"] >= train_end) & (df["date"] < pred_end)].copy()

    return train_df, test_df, train_end, pred_end


def create_sequences(data: np.ndarray, seq_length: int, target_idx: int):
    """Opret sliding window sekvenser med target."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
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
        return self.fc(lstm_out[:, -1, :])


def train_model(model, train_loader, val_loader, device):
    """Træn modellen med early stopping og cosine LR-scheduler.
    Returnerer (model, actual_epochs, stopped_early, best_val_loss, best_train_loss).
    Re-raiser KeyboardInterrupt ved Ctrl+C."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss   = float("inf")
    best_train_loss = float("inf")
    patience_counter = 0
    best_state      = None
    actual_epochs   = 0
    stopped_early   = False

    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch), y_batch).item()

            train_loss /= len(train_loader)
            val_loss   /= len(val_loader)
            scheduler.step()

            actual_epochs = epoch + 1

            if actual_epochs % 10 == 0:
                print(f"Epoch {actual_epochs}/{EPOCHS} – Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss   = val_loss
                best_train_loss = train_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            elif USE_EARLY_STOPPING:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping ved epoch {actual_epochs}")
                    stopped_early = True
                    break

    except KeyboardInterrupt:
        print("\nAfbrudt af bruger (Ctrl+C) – gemmer bedste model...")
        if best_state is not None:
            model.load_state_dict(best_state)
        raise

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, actual_epochs, stopped_early, best_val_loss, best_train_loss


def evaluate_model(model, test_loader, device):
    """Evaluer modellen på testdata. Returnerer forudsigelser og faktiske værdier (skaleret [0,1])."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device))
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    predictions = np.array(predictions)
    actuals     = np.array(actuals)
    metrics = {
        "MSE":  float(np.mean((predictions - actuals) ** 2)),
        "MAE":  float(np.mean(np.abs(predictions - actuals))),
        "MAPE": float(np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100),
    }
    return predictions, actuals, metrics


def orig_scale_metrics(actuals_orig: np.ndarray, predictions_orig: np.ndarray) -> dict:
    """Beregn metrikker i original priskala (USD)."""
    return {
        "MSE":  float(np.mean((predictions_orig - actuals_orig) ** 2)),
        "MAE":  float(np.mean(np.abs(predictions_orig - actuals_orig))),
        "MAPE": float(np.mean(np.abs((actuals_orig - predictions_orig) / (actuals_orig + 1e-8))) * 100),
    }


def save_results_md(
    actuals: np.ndarray,
    predictions: np.ndarray,
    test_dates,
    metrics: dict,
    timestamp: str,
    save_path: Path,
    train_end,
    pred_end,
    plot_filename: str = "",
    actual_epochs: int | None = None,
    train_loss: float | None = None,
    test_mse: float | None = None,
):
    """Gem resultater til en Markdown-fil. actuals/predictions er i original priskala (USD)."""
    end_actual      = float(actuals[-1])
    end_predicted   = float(predictions[-1])
    end_error_pct   = abs(end_actual - end_predicted) / end_actual * 100
    first_actual    = float(actuals[0])
    first_predicted = float(predictions[0])
    first_error_pct = abs(first_actual - first_predicted) / first_actual * 100

    errors_pct    = np.abs(actuals - predictions) / (actuals + 1e-8) * 100
    avg_error_pct = float(np.mean(errors_pct))
    max_error_pct = float(np.max(errors_pct))
    min_error_pct = float(np.min(errors_pct))

    gen_time     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plot_section = f"\n## Visualisering\n\n![Faktisk vs. forudsigelse]({plot_filename})\n\n" if plot_filename else ""

    train_val_section = ""
    if train_loss is not None or test_mse is not None:
        train_str = f"{train_loss:.6f}" if train_loss is not None else "N/A"
        test_str  = f"{test_mse:.6f}"  if test_mse  is not None else "N/A"
        train_val_section = f"""

"""

    md = f"""# Træningsresultater – {STOCK_NAME} aktieprisforudsigelse

**Genereret:** {gen_time}
{plot_section}
{train_val_section}
## Slutpris (sidste dag i testperioden)

| | Pris (USD) |
|---|---|
| **Forudsigelse** | {end_predicted:.2f} |
| **Faktisk** | {end_actual:.2f} |
| **Fejl i %** | {end_error_pct:.2f}% |

## Første dag i testperioden

| | Pris (USD) |
|---|---|
| **Forudsigelse** | {first_predicted:.2f} |
| **Faktisk** | {first_actual:.2f} |
| **Fejl i %** | {first_error_pct:.2f}% |

## Metrikker (hele testperioden – original priskala, USD)

| Metrik | Værdi |
|---|---|
| MSE | {metrics['MSE']:.4f} |
| MAE | {metrics['MAE']:.4f} USD |
| MAPE | {metrics['MAPE']:.2f}% |
| Gns. fejl i % | {avg_error_pct:.2f}% |
| Maks. fejl i % | {max_error_pct:.2f}% |
| Min. fejl i % | {min_error_pct:.2f}% |

## Træningsindstillinger

- Træningsperiode: første {TRAINING_YEARS} år (til {str(train_end)[:10]})
- Forudsigelsesperiode: {str(train_end)[:10]} til {str(pred_end)[:10]} ({PREDICTION_YEARS} år)
- Epoker (kørt): {actual_epochs if actual_epochs is not None else EPOCHS}
- Learning rate: {LEARNING_RATE}
- Batch size: {BATCH_SIZE}
- Sekvenslængde: {SEQUENCE_LENGTH} dage
- Antal forudsigelser: {len(predictions)}
- Testperiode (data): {str(test_dates[0])[:10]} til {str(test_dates[-1])[:10]}

## Træningsloss (bedste model, skaleret [0,1])

| Loss | Værdi |
|---|---|
| **Train Loss** | {train_str} |
| **Test MSE** | {test_str} |
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(md, encoding="utf-8")
    print(f"Resultater gemt til {save_path}")


def plot_results(actuals, predictions, test_dates, save_path: str, train_end, pred_end):
    """Visualiser faktisk vs. forudsigelse."""
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actuals,     label="Faktisk pris", alpha=0.8)
    plt.plot(test_dates, predictions, label="Forudsigelse",  alpha=0.8)
    plt.xlabel("Dato")
    plt.ylabel("Close-pris (USD)")
    plt.title(f"{STOCK_NAME}: Faktisk vs. forudsigelse ({str(train_end)[:10]} til {str(pred_end)[:10]})")
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
    print(f"Indstillinger: EPOCHS={EPOCHS}, LR={LEARNING_RATE}, BATCH={BATCH_SIZE}, SEQ={SEQUENCE_LENGTH}")

    # 1. Load og split baseret på TRAINING_YEARS
    train_df, test_df, train_end, pred_end = load_and_prepare_data(csv_path)
    print(f"Træningsdata: {len(train_df)} rækker (til {str(train_end)[:10]}, første {TRAINING_YEARS} år)")
    print(f"Testdata (facit): {len(test_df)} rækker ({str(train_end)[:10]} til {str(pred_end)[:10]})")

    if len(test_df) == 0:
        print(f"Fejl: Ingen testdata – CSV indeholder ikke data efter de første {TRAINING_YEARS} år.")
        return

    # 2. Features og target
    feature_cols = ["open", "high", "low", "close", "volume"]
    target_col   = "close"
    target_idx   = feature_cols.index(target_col)

    # 3. Normalisering (fit kun på træningsdata)
    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols].values)
    test_scaled  = scaler.transform(test_df[feature_cols].values)

    # 4. Kombiner for sekvenser der krydser split (test behøver SEQUENCE_LENGTH dages historik bagfra)
    full_scaled   = np.vstack([train_scaled, test_scaled])
    full_dates    = pd.concat([train_df["date"], test_df["date"]]).reset_index(drop=True)
    train_end_idx = len(train_df)

    # 5. Opret sekvenser
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, target_idx)

    val_size         = max(1, int(len(X_train) * VALIDATION_SPLIT))
    X_val,   y_val   = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Test: brug de sidste SEQUENCE_LENGTH dage fra træning som kontekst
    test_context = full_scaled[train_end_idx - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_context, SEQUENCE_LENGTH, target_idx)
    test_dates     = full_dates.iloc[train_end_idx:].values

    print(f"Træningssekvenser: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 6. DataLoaders
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(StockDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(StockDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    # 7. Træning – genstarter ved early stopping, stopper kun ved Ctrl+C
    model_path              = Path(__file__).parent / "stock_model.pt"
    best_overall_test_mse   = float("inf")
    best_overall_train_loss = None

    if model_path.exists():
        try:
            prev = torch.load(model_path, map_location="cpu", weights_only=False)
            best_overall_test_mse   = prev.get("test_mse",   float("inf"))
            best_overall_train_loss = prev.get("train_loss")
        except Exception:
            pass

    def save_model_if_better(current_model, test_mse, train_loss=None):
        nonlocal best_overall_test_mse, best_overall_train_loss
        if test_mse < best_overall_test_mse:
            best_overall_test_mse = test_mse
            if train_loss is not None:
                best_overall_train_loss = train_loss
            torch.save(
                {
                    "model_state":  current_model.state_dict(),
                    "scaler_min":   scaler.data_min_,
                    "scaler_max":   scaler.data_max_,
                    "feature_cols": feature_cols,
                    "seq_length":   SEQUENCE_LENGTH,
                    "test_mse":     test_mse,
                    "train_loss":   best_overall_train_loss,
                },
                model_path,
            )
            print(f"  → Model GEMT (test MSE: {test_mse:.6f})")
        else:
            print(f"  → Model IKKE gemt (test MSE: {test_mse:.6f} – ikke bedre)")

    model           = None
    run_number      = 0
    actual_epochs   = 0
    best_train_loss = None

    try:
        while True:
            run_number += 1
            model = LSTMModel(input_size=len(feature_cols)).to(device)
            print(f"\nTræningsrunde {run_number}... (Ctrl+C for at stoppe)")
            model, actual_epochs, stopped_early, _, best_train_loss = train_model(
                model, train_loader, val_loader, device
            )
            _, _, metrics = evaluate_model(model, test_loader, device)
            save_model_if_better(model, metrics["MSE"], best_train_loss)
            status = "Early stopping" if stopped_early else "Alle epoker færdige"
            print(f"{status} – genstarter... (Ctrl+C for at stoppe)")

    except KeyboardInterrupt:
        print("\n\nStoppet af bruger (Ctrl+C).")
        if model is not None:
            _, _, metrics = evaluate_model(model, test_loader, device)
            save_model_if_better(model, metrics["MSE"])
        else:
            print("Ingen model at gemme.")
            return

    if not model_path.exists():
        print("Fejl: Ingen gemt model fundet.")
        return

    # 8. Indlæs bedste model og evaluer
    print("\nIndlæser bedste model og evaluerer på testdata...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = LSTMModel(input_size=len(feature_cols)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    saved_train_loss = checkpoint.get("train_loss")
    saved_test_mse   = checkpoint.get("test_mse")

    predictions_scaled, actuals_scaled, _ = evaluate_model(model, test_loader, device)

    # 9. Transformér tilbage til original priskala (USD)
    close_min = float(scaler.data_min_[target_idx])
    close_max = float(scaler.data_max_[target_idx])
    actuals_orig     = actuals_scaled     * (close_max - close_min) + close_min
    predictions_orig = predictions_scaled * (close_max - close_min) + close_min

    metrics_orig = orig_scale_metrics(actuals_orig, predictions_orig)

    print("\n--- Resultater (original priskala, USD) ---")
    print(f"MSE:  {metrics_orig['MSE']:.4f}")
    print(f"MAE:  {metrics_orig['MAE']:.4f} USD")
    print(f"MAPE: {metrics_orig['MAPE']:.2f}%")

    # 10. Gem plot og rapport
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M")
    date_folder = datetime.now().strftime("%Y-%m-%d")
    output_dir  = Path(__file__).parent / "Prediction_pictures" / date_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"prediction_plot_{timestamp}.png"
    plot_results(actuals_orig, predictions_orig, test_dates, str(plot_path), train_end, pred_end)

    results_path = output_dir / f"resultater_{timestamp}.md"
    save_results_md(
        actuals_orig, predictions_orig, test_dates, metrics_orig,
        timestamp, results_path, train_end, pred_end,
        plot_path.name, actual_epochs, saved_train_loss, saved_test_mse,
    )


if __name__ == "__main__":
    main()
