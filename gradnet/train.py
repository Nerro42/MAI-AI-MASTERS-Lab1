from __future__ import annotations

from typing import Callable

import numpy as np

from .data import Pack
from .nn import Module
from .optim import _Base, clip_grad_norm
from .tensor import Tensor

MetricFn = Callable[[Tensor, np.ndarray], float]
LossFn = Callable[[Tensor, np.ndarray], Tensor]


class Loop:
    def __init__(
        self,
        model: Module,
        loss: LossFn,
        opt: _Base,
        metric: MetricFn | None = None,
        clip: float | None = None,
    ):
        self.model = model
        self.loss = loss
        self.opt = opt
        self.metric = metric
        self.clip = clip

    def _pass(self, pack: Pack, batch: int, train: bool, seed: int) -> dict[str, float]:
        self.model.train_mode(train)
        loss_sum = 0.0
        metric_sum = 0.0
        seen = 0
        for xb, yb in pack.batches(batch, shuffle=train, seed=seed):
            if train:
                self.model.zero_grad()
            pred = self.model(xb)
            loss = self.loss(pred, yb)
            if train:
                loss.backward()
                if self.clip is not None:
                    clip_grad_norm(self.model.parameters(), self.clip)
                self.opt.step()
            n = int(len(xb))
            loss_sum += float(np.asarray(loss.data)) * n
            seen += n
            if self.metric is not None:
                metric_sum += self.metric(pred, yb) * n
        out = {"loss": loss_sum / seen if seen else float("nan")}
        if self.metric is not None:
            out["metric"] = metric_sum / seen if seen else float("nan")
        return out

    def run(
        self,
        train: Pack,
        val: Pack | None = None,
        epochs: int = 20,
        batch: int = 32,
        log_every: int = 1,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
        }
        for epoch in range(1, epochs + 1):
            tr = self._pass(train, batch, train=True, seed=epoch)
            history["train_loss"].append(tr["loss"])
            history["train_metric"].append(tr.get("metric", float("nan")))

            if val is not None and len(val) > 0:
                va = self._pass(val, batch, train=False, seed=0)
                history["val_loss"].append(va["loss"])
                history["val_metric"].append(va.get("metric", float("nan")))
            else:
                history["val_loss"].append(float("nan"))
                history["val_metric"].append(float("nan"))

            if verbose and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                line = f"[{epoch:03d}/{epochs}] loss={tr['loss']:.4f}"
                if self.metric is not None:
                    line += f" metric={tr.get('metric', float('nan')):.3f}"
                if val is not None:
                    line += f"  val={history['val_loss'][-1]:.4f}"
                    if self.metric is not None:
                        line += f" val_metric={history['val_metric'][-1]:.3f}"
                print(line)
        return history
