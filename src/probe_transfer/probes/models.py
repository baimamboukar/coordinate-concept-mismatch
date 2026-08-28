from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FrozenPreprocessor:
    mean: np.ndarray
    scale: float

    @classmethod
    def fit(cls, activations: np.ndarray) -> "FrozenPreprocessor":
        values = np.asarray(activations, dtype=np.float32)
        mean = values.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.var(values, axis=0))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Activation scale must be finite and positive.")
        return cls(mean=mean, scale=scale)

    def transform(self, activations: np.ndarray) -> np.ndarray:
        values = np.asarray(activations, dtype=np.float32)
        return (values - self.mean) / self.scale


@dataclass
class LinearProbe:
    preprocessor: FrozenPreprocessor
    estimator: LogisticRegression
    validation_auroc: float
    c: float
    iterations: int

    def scores(self, activations: np.ndarray) -> np.ndarray:
        values = self.preprocessor.transform(activations)
        return self.estimator.decision_function(values)


def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    c_values: Iterable[float],
    max_iter: int = 5000,
) -> LinearProbe:
    preprocessor = FrozenPreprocessor.fit(train_x)
    transformed_train = preprocessor.transform(train_x)
    transformed_validation = preprocessor.transform(validation_x)
    candidates = []

    for c in c_values:
        estimator = LogisticRegression(C=float(c), max_iter=max_iter, solver="lbfgs")
        with catch_warnings(record=True) as warnings:
            simplefilter("always", ConvergenceWarning)
            estimator.fit(transformed_train, train_y)
        if any(issubclass(item.category, ConvergenceWarning) for item in warnings):
            raise RuntimeError(
                f"Linear probe with C={float(c)} did not converge within {max_iter} iterations."
            )
        score = float(
            roc_auc_score(validation_y, estimator.decision_function(transformed_validation))
        )
        candidates.append((score, -float(c), estimator, float(c), int(estimator.n_iter_.max())))

    if not candidates:
        raise ValueError("At least one linear-probe C value is required.")
    score, _, estimator, c, iterations = max(candidates, key=lambda item: (item[0], item[1]))
    return LinearProbe(preprocessor, estimator, score, c, iterations)


class CPDegree2(nn.Module):
    def __init__(self, input_size: int, rank: int) -> None:
        super().__init__()
        self.left = nn.Linear(input_size, rank)
        self.right = nn.Linear(input_size, rank)
        self.alpha = nn.Parameter(torch.ones(rank))
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        quadratic = (self.left(inputs) * self.right(inputs) * self.alpha).sum(dim=-1)
        return quadratic + self.linear(inputs).squeeze(-1)


class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


@dataclass
class NeuralProbe:
    preprocessor: FrozenPreprocessor
    model: nn.Module
    validation_auroc: float
    epochs: int

    def scores(self, activations: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        values = torch.from_numpy(self.preprocessor.transform(activations))
        loader = DataLoader(TensorDataset(values), batch_size=batch_size)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outputs)


def train_neural_probe(
    model_factory: Callable[[int], nn.Module],
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str = "cpu",
) -> NeuralProbe:
    torch.manual_seed(seed)
    preprocessor = FrozenPreprocessor.fit(train_x)
    transformed_train = torch.from_numpy(preprocessor.transform(train_x))
    labels = torch.as_tensor(train_y, dtype=torch.float32)
    validation_values = torch.from_numpy(preprocessor.transform(validation_x)).to(device)
    model = model_factory(transformed_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(transformed_train, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_score = float("-inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(validation_values).cpu().numpy()
        validation_auroc = float(roc_auc_score(validation_y, scores))
        if validation_auroc > best_score:
            best_score = validation_auroc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Neural probe training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return NeuralProbe(preprocessor, model, best_score, best_epoch)
