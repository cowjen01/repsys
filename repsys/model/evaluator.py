import numpy as np
import matplotlib.pyplot as plt

# from repsys.metrics import recall


class ModelEvaluator:
    def __init__(self) -> None:
        self.metrics = {
            # "Recall@5": {"method": self.recall, "args": {"k": 5}},
            # "Recall@20": {"method": self.recall, "args": {"k": 20}},
            # "Recall@50": {"method": self.recall, "args": {"k": 50}},
            # "NCDG@100": {"method": self.ndcg, "args": {"k": 100}},
        }
        self.results = {}

    # def get_recall(self, X_pred, X_true, k=50):
    #     # TODO use jax variant of argpartition once it will be implemented
    #     idx = np.argpartition(-X_pred, k, axis=1)[:, :k]
    #     return recall(X_pred, X_true, idx, k).block_until_ready()

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
                X_pred, heldout_batch.toarray(), **metric[1].get("args")
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
