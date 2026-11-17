import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn


def _get_init_scale(in_features: int, init_method: str) -> float:
    if init_method == "he":
        return float(np.sqrt(2.0 / in_features))
    if init_method == "normal_001":
        return 0.01
    if init_method == "std_1":
        return 1.0
    if init_method == "broken_large":
        return 5.0
    raise ValueError(f"Nieznana metoda inicjalizacji: {init_method}")


class PyTorchMLP(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        alpha: float = 0.01,
        epochs: int = 100,
        batch_size: int | None = 32,
        optimizer: str = "sgd",
        init_method: str = "he",
        random_state: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.device = torch.device(device)

        modules: list[nn.Module] = []
        for i in range(len(layer_dims) - 2):
            linear = nn.Linear(layer_dims[i], layer_dims[i + 1], dtype=torch.float64)
            self._init_linear(linear, init_method, random_state + i)
            modules.append(linear)
            modules.append(nn.ReLU())

        out_linear = nn.Linear(layer_dims[-2], layer_dims[-1], dtype=torch.float64)
        self._init_linear(out_linear, init_method, random_state + len(layer_dims))
        modules.append(out_linear)

        self.net = nn.Sequential(*modules)
        self.to(self.device)

        self.history: dict[str, list[float]] = {
            "loss": [],
            "test_loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

    @staticmethod
    def _init_linear(layer: nn.Linear, init_method: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        in_features = layer.in_features
        out_features = layer.out_features
        scale = _get_init_scale(in_features, init_method)

        w_np = rng.normal(loc=0.0, scale=scale, size=(in_features, out_features))
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(w_np.T))
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.alpha)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.alpha, betas=(0.9, 0.999), eps=1e-8)
        raise ValueError(f"Nieznany optymalizator: {self.optimizer_name}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> "PyTorchMLP":
        if (X_eval is None) != (y_eval is None):
            raise ValueError("X_eval i y_eval muszą być przekazane razem.")

        n_samples = X.shape[0]
        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=self.device)

        X_eval_tensor = (
            torch.tensor(X_eval, dtype=torch.float64, device=self.device) if X_eval is not None else None
        )
        y_eval_tensor = (
            torch.tensor(y_eval.reshape(-1, 1), dtype=torch.float64, device=self.device) if y_eval is not None else None
        )

        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = self._build_optimizer()
        rng = np.random.default_rng(42)

        self.history = {
            "loss": [],
            "test_loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        for _ in range(self.epochs):
            self.train()
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            if self.batch_size is None:
                batch_slices = [indices]
            else:
                batch_slices = [indices[i : i + self.batch_size] for i in range(0, n_samples, self.batch_size)]

            for idx in batch_slices:
                X_batch = X_tensor[idx]
                y_batch = y_tensor[idx]

                optimizer.zero_grad()
                logits = self(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                train_logits = self(X_tensor)
                train_loss = float(criterion(train_logits, y_tensor).item())
                train_probs = torch.sigmoid(train_logits).cpu().numpy().ravel()
                y_pred = (train_probs >= 0.5).astype(int)

                self.history["loss"].append(train_loss)
                self.history["accuracy"].append(float(accuracy_score(y, y_pred)))
                self.history["f1"].append(float(f1_score(y, y_pred, zero_division=0)))
                self.history["precision"].append(float(precision_score(y, y_pred, zero_division=0)))
                self.history["recall"].append(float(recall_score(y, y_pred, zero_division=0)))

                if X_eval_tensor is not None and y_eval_tensor is not None:
                    eval_logits = self(X_eval_tensor)
                    eval_loss = float(criterion(eval_logits, y_eval_tensor).item())
                    self.history["test_loss"].append(eval_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
            logits = self(X_tensor)
            return torch.sigmoid(logits).cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
