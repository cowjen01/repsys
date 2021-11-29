import numpy as np
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp


@jit
def recall_jax(X_pred, X_true, argsort_indices, k):
    X_pred_binary = jnp.zeros_like(X_pred, dtype=bool)
    X_pred_binary = X_pred_binary.at[
        jnp.arange(X_pred.shape[0])[:, jnp.newaxis], argsort_indices
    ].set(True)
    X_true_binary = X_true > 0
    tmp = (jnp.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        jnp.float32
    )
    recall = tmp / jnp.minimum(k, X_true_binary.sum(axis=1))
    return recall


class ModelEvaluator:
    def __init__(self) -> None:
        self.metrics = {
            # "Recall@5": {"method": self.recall, "args": {"k": 5}},
            # "Recall@20": {"method": self.recall, "args": {"k": 20}},
            "Recall@50": {"method": self.recall, "args": {"k": 50}},
            # "NCDG@100": {"method": self.ndcg, "args": {"k": 100}},
        }
        self.results = {}

    def recall(self, X_pred, X_true, k=100):
        idx = np.argpartition(-X_pred, k, axis=1)
        return recall_jax(X_pred, X_true.toarray(), idx[:, :k], k)

    def ndcg(self, X_pred, heldout_batch, k=100):
        batch_users = X_pred.shape[0]
        idx_topk_part = np.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[
            np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]
        ]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[
            np.arange(batch_users)[:, np.newaxis], idx_part
        ]
        tp = 1.0 / np.log2(np.arange(2, k + 2))

        DCG = (
            heldout_batch[
                np.arange(batch_users)[:, np.newaxis], idx_topk
            ].toarray()
            * tp
        ).sum(axis=1)
        IDCG = np.array(
            [(tp[: min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)]
        )
        return DCG / IDCG

    def evaluate_model(self, model, X, heldout_batch):
        X_pred = model.predict(X)
        self.results[model.name()] = {}

        for metric in self.metrics.items():
            self.results[model.name()][metric[0]] = metric[1].get("method")(
                X_pred, heldout_batch, **metric[1].get("args")
            )

    def print_results(self):
        for i, model in enumerate(self.results.keys()):
            print(f"Model: {model}")
            for result in self.results[model].items():
                print(
                    f"{result[0]}=%.5f (%.5f)"
                    % (
                        np.mean(result[1]),
                        np.std(result[1]) / np.sqrt(len(result[1])),
                    )
                )

            if i < len(self.results.keys()) - 1:
                print("-------------")

    def plot_distributions(self):
        for model in self.results.keys():

            fig, axs = plt.subplots(
                1, len(self.metrics.keys()), figsize=(15, 4)
            )

            fig.suptitle(f"Model {model}", fontsize=16)

            for i, result in enumerate(self.results[model].items()):
                ax = axs[i]
                ax.hist(result[1])
                ax.set_title(result[0])

            fig.tight_layout()
            plt.show()
