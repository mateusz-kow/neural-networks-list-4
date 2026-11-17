# SIECI NEURONOWE – ćwiczenie

Implementacja propagacji wstecznej na zbiorze Heart Disease (PyTorch)

Repozytorium: [https://github.com/mateusz-kow/neural-networks-list-4](https://github.com/mateusz-kow/neural-networks-list-4)

Autor: Mateusz Kowalski 280543

---

### Wyniki eksperymentów

#### Wpływ wymiarowości warstwy ukrytej

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part1_hidden_size_loss.png) | ![](plots/part1_hidden_size_loss.png) |

#### Lista 3:

Hidden size: Mała (4)                  | Test Loss: 0.5859 | Acc: 0.885 | Prec: 0.957 | Rec: 0.786 | F1: 0.863

Hidden size: Równa cechom (25)         | Test Loss: 0.4825 | Acc: 0.885 | Prec: 0.889 | Rec: 0.857 | F1: 0.873

Hidden size: Duża (128)                | Test Loss: 0.3539 | Acc: 0.902 | Prec: 0.893 | Rec: 0.893 | F1: 0.893

#### Lista 4:

Hidden size: Mała (4)                  | Test Loss: 0.5859 | Acc: 0.885 | Prec: 0.957 | Rec: 0.786 | F1: 0.863

Hidden size: Równa cechom (25)         | Test Loss: 0.4825 | Acc: 0.885 | Prec: 0.889 | Rec: 0.857 | F1: 0.873

Hidden size: Duża (128)                | Test Loss: 0.3539 | Acc: 0.902 | Prec: 0.893 | Rec: 0.893 | F1: 0.893

#### Wpływ normalizacji danych

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part1_normalization_loss.png) | ![](plots/part1_normalization_loss.png) |

#### Lista 3:

Dane: Znormalizowane       | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Dane: Nieznormalizowane    | Test Loss: 0.5500 | Acc: 0.721 | Prec: 0.704 | Rec: 0.679 | F1: 0.691

#### Lista 4:

Dane: Znormalizowane       | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Dane: Nieznormalizowane    | Test Loss: 0.5500 | Acc: 0.721 | Prec: 0.704 | Rec: 0.679 | F1: 0.691

#### Wpływ głębokości sieci (liczba warstw ukrytych)

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part1_layers_loss.png) | ![](plots/part1_layers_loss.png) |

#### Lista 3:

Arch: 1 warstwa ukryta [16]        | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Arch: 2 warstwy ukryte [16, 16]    | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

Arch: 3 warstwy ukryte [16, 16, 16] | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

#### Lista 4:

Arch: 1 warstwa ukryta [16]        | Test Loss: 0.6044 | Acc: 0.836 | Prec: 0.950 | Rec: 0.679 | F1: 0.792

Arch: 2 warstwy ukryte [16, 16]    | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

Arch: 3 warstwy ukryte [16, 16, 16] | Test Loss: 0.6898 | Acc: 0.541 | Prec: 0.000 | Rec: 0.000 | F1: 0.000

#### Wpływ paczkowania

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part2_batches_loss.png) | ![](plots/part2_batches_loss.png) |

#### Lista 3:

Tryb: Pełny zbiór (Batch GD)           | Test Loss: 0.6791 | Acc: 0.574 | Prec: 0.542 | Rec: 0.464 | F1: 0.500

Tryb: Pojedyncze przykłady (SGD, b=1)  | Test Loss: 1.1517 | Acc: 0.852 | Prec: 0.806 | Rec: 0.893 | F1: 0.847

Tryb: Paczki (Mini-batch, b=32)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

#### Lista 4:

Tryb: Pełny zbiór (Batch GD)           | Test Loss: 0.6791 | Acc: 0.574 | Prec: 0.542 | Rec: 0.464 | F1: 0.500

Tryb: Pojedyncze przykłady (SGD, b=1)  | Test Loss: 1.1517 | Acc: 0.852 | Prec: 0.806 | Rec: 0.893 | F1: 0.847

Tryb: Paczki (Mini-batch, b=32)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

#### Wpływ rozkładu początkowego

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part2_init_loss.png) | ![](plots/part2_init_loss.png) |

Na tym wykresie widać największy rozjazd wyników. Jest to spowodowane tym, że w liście 3 ucinałem skraje wartości prawdopodobieństw (< 1e-15 lub > 1.0 - 1e-15), czego nie udało mi się powtórzyć w pytorch.

#### Lista 3:

Init: Inicjalizacja He (optymalna)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Init: Odchylenie standardowe = 1.0        | Test Loss: 0.8317 | Acc: 0.787 | Prec: 0.742 | Rec: 0.821 | F1: 0.780

Init: Zepsuta (duże odchylenie std=5.0)   | Test Loss: 5.8577 | Acc: 0.803 | Prec: 0.767 | Rec: 0.821 | F1: 0.793

#### Lista 4:

Init: Inicjalizacja He (optymalna)        | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Init: Odchylenie standardowe = 1.0        | Test Loss: 0.8317 | Acc: 0.787 | Prec: 0.742 | Rec: 0.821 | F1: 0.780

Init: Zepsuta (duże odchylenie std=5.0)   | Test Loss: 44.4030 | Acc: 0.738 | Prec: 0.676 | Rec: 0.821 | F1: 0.742

#### Wpływ optymalizacji przez prosty gradient vs. ADAM

| Lista 3 (NumPy) | Lista 4 (PyTorch) |
| :---: | :---: |
| ![](plots_l3/part2_optimizer_loss.png) | ![](plots/part2_optimizer_loss.png) |

#### Lista 3:

Optymalizator: Prosty SGD (alpha=0.01)             | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Optymalizator: Optymalizator ADAM (alpha=0.01)     | Test Loss: 0.9795 | Acc: 0.852 | Prec: 0.788 | Rec: 0.929 | F1: 0.852

#### Lista 4:

Optymalizator: Prosty SGD (alpha=0.01)             | Test Loss: 0.3004 | Acc: 0.902 | Prec: 0.867 | Rec: 0.929 | F1: 0.897

Optymalizator: Optymalizator ADAM (alpha=0.01)     | Test Loss: 0.9796 | Acc: 0.852 | Prec: 0.788 | Rec: 0.929 | F1: 0.852


### Jak doprowadzono do spójnych wyików między listą 3 a listą 4?

1./ Zastąpienie domyślnego generatora rozkładu wag pytorch własnym oraz użycie seeda do generowania liczb losowych.

```python
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
```

2./ Zastosowanie podwójnej precyzji zmiennoprzecinkowej

Polegało to na nadpisaniu domyślnego dla pytorch typu zmiennej float32 typem foat64, którego domyślnie używa numpy

3./ Funkcja straty

Zastosowano funkcję `nn.BCEWithLogitsLoss(reduction="mean")`, której gradient jest kompatybilny z gradientem z listy 3

4./ Ręczne losowanie kolejności tensorów w zbiorze treningowym

Wykorzystano lekko zmodyfikowany kod z listy 3, z tym samym seedem generatora liczb losowych.

### Baseline CIFAR-10

![](plots/cifar10_baseline_loss.png)

Epoka 01/15 (13.0s) | Train Loss: 1.6670 | Test Loss: 1.4797 | Test Acc: 0.4746 | Macro F1: 0.4676

Epoka 02/15 (9.3s) | Train Loss: 1.4797 | Test Loss: 1.3684 | Test Acc: 0.5129 | Macro F1: 0.5067

Epoka 03/15 (9.0s) | Train Loss: 1.3948 | Test Loss: 1.3463 | Test Acc: 0.5183 | Macro F1: 0.5123

Epoka 04/15 (9.4s) | Train Loss: 1.3361 | Test Loss: 1.3128 | Test Acc: 0.5314 | Macro F1: 0.5247

Epoka 05/15 (11.3s) | Train Loss: 1.2821 | Test Loss: 1.2796 | Test Acc: 0.5443 | Macro F1: 0.5392

Epoka 06/15 (9.1s) | Train Loss: 1.2364 | Test Loss: 1.2478 | Test Acc: 0.5556 | Macro F1: 0.5517

Epoka 07/15 (8.7s) | Train Loss: 1.1894 | Test Loss: 1.2353 | Test Acc: 0.5580 | Macro F1: 0.5549

Epoka 08/15 (9.3s) | Train Loss: 1.1428 | Test Loss: 1.2227 | Test Acc: 0.5673 | Macro F1: 0.5640

Epoka 09/15 (8.2s) | Train Loss: 1.1020 | Test Loss: 1.2071 | Test Acc: 0.5663 | Macro F1: 0.5633

Epoka 10/15 (8.2s) | Train Loss: 1.0582 | Test Loss: 1.1980 | Test Acc: 0.5736 | Macro F1: 0.5706

Epoka 11/15 (8.4s) | Train Loss: 1.0215 | Test Loss: 1.1851 | Test Acc: 0.5791 | Macro F1: 0.5765

Epoka 12/15 (8.2s) | Train Loss: 0.9841 | Test Loss: 1.1830 | Test Acc: 0.5820 | Macro F1: 0.5792

Epoka 13/15 (8.2s) | Train Loss: 0.9611 | Test Loss: 1.1787 | Test Acc: 0.5814 | Macro F1: 0.5775

Epoka 14/15 (8.1s) | Train Loss: 0.9396 | Test Loss: 1.1758 | Test Acc: 0.5830 | Macro F1: 0.5802

Epoka 15/15 (8.3s) | Train Loss: 0.9330 | Test Loss: 1.1750 | Test Acc: 0.5841 | Macro F1: 0.5821

#### Wnioski

Loss pozostał dość duży pomimo zastosowania 15 epok. Świadczy to o tym, że liczba epok jest za mała, albo model jest zbyt mały.

---

Wykorzystanie generatywnej sztucznej inteligencji w ramach zadania ogranicało się głównie do wygenerowania kodu tworzącego wykresy oraz liczącego metryki a także naprawy błędów wykorzystania biblioteki pytorch.
