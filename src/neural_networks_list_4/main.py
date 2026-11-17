import os
import shutil
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .cifar10 import train_cifar10_baseline
from .data import prepare_data
from .model import PyTorchMLP
from .utils import plot_learning_curves

EPOCHS = 1000
EPOCHS_PART2 = 100


def _print_metrics(prefix: str, history: dict, y_true, y_pred) -> None:
    print(
        f"{prefix} | Test Loss: {history['test_loss'][-1]:.4f} | "
        f"Acc: {accuracy_score(y_true, y_pred):.3f} | Prec: {precision_score(y_true, y_pred, zero_division=0):.3f} | "
        f"Rec: {recall_score(y_true, y_pred, zero_division=0):.3f} | F1: {f1_score(y_true, y_pred, zero_division=0):.3f}"
    )


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    X_train_norm, X_test_norm, y_train, y_test = prepare_data(normalize=True)
    n_features = X_train_norm.shape[1]

    print(f"Liczba cech po przetworzeniu: {n_features}")

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 1: Wymiarowość warstwy ukrytej
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 1: Wymiarowość warstwy ukrytej (PyTorch)")
    print("=" * 80)

    hidden_sizes = {
        "Mała (4)": 4,
        f"Równa cechom ({n_features})": n_features,
        "Duża (128)": 128,
    }
    results_hidden = {}

    for label, h_size in hidden_sizes.items():
        model = PyTorchMLP(
            layer_dims=[n_features, h_size, 1],
            alpha=0.01,
            epochs=EPOCHS,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_train_norm, y_train, X_eval=X_test_norm, y_eval=y_test)
        results_hidden[label] = model.history
        _print_metrics(f"Hidden size: {label:25s}", model.history, y_test, model.predict(X_test_norm))

    plot_learning_curves(
        results_hidden,
        "Wpływ wymiarowości warstwy ukrytej na funkcję kosztu (PyTorch)",
        "plots/part1_hidden_size_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 2: Normalizacja danych
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 2: Normalizacja danych (PyTorch)")
    print("=" * 80)

    X_train_unnorm, X_test_unnorm, _, _ = prepare_data(normalize=False)
    results_norm = {}

    data_variants = {
        "Znormalizowane": (X_train_norm, X_test_norm),
        "Nieznormalizowane": (X_train_unnorm, X_test_unnorm),
    }

    for label, (X_tr, X_te) in data_variants.items():
        model = PyTorchMLP(
            layer_dims=[n_features, 16, 1],
            alpha=0.01,
            epochs=EPOCHS,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_tr, y_train, X_eval=X_te, y_eval=y_test)
        results_norm[label] = model.history
        _print_metrics(f"Dane: {label:20s}", model.history, y_test, model.predict(X_te))

    plot_learning_curves(
        results_norm,
        "Wpływ normalizacji danych na zbieżność modelu (PyTorch)",
        "plots/part1_normalization_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 1 - EKSPERYMENT 3: Różna liczba warstw (głębokość sieci)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 1 - EKSPERYMENT 3: Różna liczba warstw (PyTorch)")
    print("=" * 80)

    architecture_variants = {
        "1 warstwa ukryta [16]": [n_features, 16, 1],
        "2 warstwy ukryte [16, 16]": [n_features, 16, 16, 1],
        "3 warstwy ukryte [16, 16, 16]": [n_features, 16, 16, 16, 1],
    }
    results_layers = {}

    for label, arch in architecture_variants.items():
        model = PyTorchMLP(
            layer_dims=arch,
            alpha=0.01,
            epochs=EPOCHS,
            batch_size=None,
            optimizer="sgd",
            init_method="normal_001",
        )
        model.fit(X_train_norm, y_train, X_eval=X_test_norm, y_eval=y_test)
        results_layers[label] = model.history
        _print_metrics(f"Arch: {label:28s}", model.history, y_test, model.predict(X_test_norm))

    plot_learning_curves(
        results_layers,
        "Wpływ liczby warstw ukrytych na zbieżność modelu (PyTorch)",
        "plots/part1_layers_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 1: Tryby paczkowania danych
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 1: Tryby paczkowania danych (PyTorch)")
    print("=" * 80)

    deep_architecture = [n_features, 16, 16, 1]
    batch_variants = {
        "Pełny zbiór (Batch GD)": None,
        "Pojedyncze przykłady (SGD, b=1)": 1,
        "Paczki (Mini-batch, b=32)": 32,
    }
    results_batches = {}

    for label, b_size in batch_variants.items():
        model = PyTorchMLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=EPOCHS_PART2,
            batch_size=b_size,
            optimizer="sgd",
            init_method="he",
        )
        model.fit(X_train_norm, y_train, X_eval=X_test_norm, y_eval=y_test)
        results_batches[label] = model.history
        _print_metrics(f"Tryb: {label:32s}", model.history, y_test, model.predict(X_test_norm))

    plot_learning_curves(
        results_batches,
        "Porównanie trybów paczkowania danych (PyTorch)",
        "plots/part2_batches_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 2: Metody inicjalizacji wag
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 2: Metody inicjalizacji wag (PyTorch)")
    print("=" * 80)

    init_variants = {
        "Inicjalizacja He (optymalna)": "he",
        "Odchylenie standardowe = 1.0": "std_1",
        "Zepsuta (duże odchylenie std=5.0)": "broken_large",
    }
    results_init = {}

    for label, init_m in init_variants.items():
        model = PyTorchMLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=EPOCHS_PART2,
            batch_size=32,
            optimizer="sgd",
            init_method=init_m,
        )
        model.fit(X_train_norm, y_train, X_eval=X_test_norm, y_eval=y_test)
        results_init[label] = model.history
        _print_metrics(f"Init: {label:35s}", model.history, y_test, model.predict(X_test_norm))

    plot_learning_curves(
        results_init,
        "Porównanie metod inicjalizacji wag (PyTorch)",
        "plots/part2_init_loss.png",
    )

    # =========================================================================
    # CZĘŚĆ 2 - EKSPERYMENT 3: Prosty SGD vs Optymalizator ADAM
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CZĘŚĆ 2 - EKSPERYMENT 3: Optymalizatory SGD oraz ADAM (PyTorch)")
    print("=" * 80)

    optimizer_variants = {
        "Prosty SGD (alpha=0.01)": "sgd",
        "Optymalizator ADAM (alpha=0.01)": "adam",
    }
    results_opt = {}

    for label, opt in optimizer_variants.items():
        model = PyTorchMLP(
            layer_dims=deep_architecture,
            alpha=0.01,
            epochs=EPOCHS_PART2,
            batch_size=32,
            optimizer=opt,
            init_method="he",
        )
        model.fit(X_train_norm, y_train, X_eval=X_test_norm, y_eval=y_test)
        results_opt[label] = model.history
        _print_metrics(f"Optymalizator: {label:35s}", model.history, y_test, model.predict(X_test_norm))

    plot_learning_curves(
        results_opt,
        "Porównanie optymalizatorów SGD oraz ADAM (PyTorch)",
        "plots/part2_optimizer_loss.png",
    )

    # =========================================================================
    # CIFAR-10 BASELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" CIFAR-10 BASELINE (MLP 3072 -> 512 -> 256 -> 10)")
    print("=" * 80)
    cifar_history = train_cifar10_baseline(epochs=15, batch_size=128, lr=1e-3)
    plot_learning_curves(
        {"CIFAR-10 MLP Baseline": cifar_history},
        "Krzywa uczenia baseline MLP na zbiorze CIFAR-10",
        "plots/cifar10_baseline_loss.png",
    )


if __name__ == "__main__":
    main()
