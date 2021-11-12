import logging
import os
from typing import Dict, Text
from sanic import Sanic
from sanic.response import json, file
import numpy as np
from sanic import exceptions
from scipy import sparse

from .models import Model
from .dataset import Dataset

logger = logging.getLogger(__name__)


def create_app(models: Dict[Text, Model], dataset: Dataset):
    app = Sanic(__name__)

    static_folder = os.path.join(os.path.dirname(__file__), "../frontend/build")
    app.static("/", static_folder)

    @app.route("/")
    async def index(request):
        return await file(f"{static_folder}/index.html")

    @app.route("/api/models")
    def get_models(request):
        return json(
            [
                {
                    "key": k,
                    "attributes": [
                        {"key": p.key, "type": p.type, "label": p.label}
                        for p in models[k].website_params()
                    ],
                }
                for k in models.keys()
            ]
        )

    @app.route("/api/predict", methods=["POST"])
    def post_prediction(request):
        user_index = request.json.get("user")
        interactions = request.json.get("interactions")
        limit = request.json.get("limit", 20)
        params = request.json.get("params", {})
        model_name = request.json.get("model")

        if (user_index is None and interactions is None) or (
            user_index is not None and interactions is not None
        ):
            raise exceptions.InvalidUsage(
                "Either the user or his interactions must be specified."
            )

        if not model_name:
            raise exceptions.InvalidUsage("Model name must be specified.")

        model = models.get(model_name)

        if not model:
            raise exceptions.NotFound(f"Model '{model_name}' was not found.")

        default_params = {
            p.key: p.default_value for p in model.website_params()
        }
        clean_params = {
            k: v
            for k, v in params.items()
            if k in default_params
        }
        predict_params = {**default_params, **clean_params}

        if user_index is not None:
            try:
                X = dataset.vad_data_tr[user_index]
            except Exception:
                raise exceptions.NotFound(f"User '{user_index}' was not found.")

        else:
            interactions = np.array(interactions)
            X = sparse.csr_matrix(
                (
                    np.ones_like(interactions),
                    (np.zeros_like(interactions), interactions),
                ),
                dtype="float64",
                shape=(1, dataset.n_items),
            )

        prediction = model.predict(X, **predict_params)

        idxs = (-prediction[0]).argsort()[:limit]

        items = dataset.items.loc[idxs]

        return json(items.to_dict("records"))

    @app.route("/api/users")
    def get_users(request):
        return json(dataset.vad_users.to_dict("records"))

    @app.route("/api/items")
    def get_items(request):
        query_str = request.args.get("query")

        if not query_str or len(query_str) == 0:
            return json([])

        df = dataset.items
        df = df[df["title"].str.contains(query_str, case=False)]

        return json(df.to_dict("records"))

    @app.route("/api/interactions")
    def get_interactions(request):
        user_index = request.args.get("user")

        if user_index is None:
            raise exceptions.InvalidUsage("User must be specified.")

        try:
            interactions = dataset.vad_data_tr[int(user_index)]
        except Exception:
            raise exceptions.NotFound(f"User '{user_index}' was not found.")

        items = dataset.items.loc[(interactions > 0).indices]

        return json(items.to_dict("records"))

    @app.listener("after_server_stop")
    def on_shutdown(app, loop):
        logger.info("Server has been shut down.")

    return app


def run_server(port, models, dataset) -> None:
    app = create_app(models, dataset)
    app.run(host="localhost", port=port)
