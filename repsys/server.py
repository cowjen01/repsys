import logging
import os
from typing import Dict, Text
from sanic import Sanic
from sanic.response import json, file
import numpy as np
from sanic import exceptions

from repsys.dataset import Dataset
from repsys.model import Model

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
        return json([m.to_dict() for m in models.values()])

    @app.route("/api/users")
    def get_users(request):
        return json(dataset.vad_users)

    @app.route("/api/items")
    def get_items(request):
        query_str = request.args.get("query")

        if not query_str or len(query_str) == 0:
            return json([])

        title_col = dataset.get_item_view_col("title")
        items = dataset.filter_items(title_col, query_str)

        return json(items.to_dict("records"))

    @app.route("/api/interactions")
    def get_interactions(request):
        user_id = request.args.get("user")

        if user_id is None:
            raise exceptions.InvalidUsage("User must be specified.")

        try:
            user_id = int(user_id)
            items = dataset.get_interacted_items(user_id)
        except Exception:
            raise exceptions.NotFound(f"User '{user_id}' was not found.")

        return json(items.to_dict("records"))

    @app.route("/api/predict", methods=["POST"])
    def post_prediction(request):
        user_id = request.json.get("user")
        interactions = request.json.get("interactions")
        limit = request.json.get("limit", 20)
        params = request.json.get("params", {})
        model_name = request.json.get("model")

        if (user_id is None and interactions is None) or (
            user_id is not None and interactions is not None
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
            p.name: p.default for p in model.web_params()
        }
        cleaned_params = {
            k: v for k, v in params.items() if k in default_params
        }
        predict_params = {**default_params, **cleaned_params}

        if user_id is not None:
            try:
                user_id = int(user_id)
                X = dataset.get_user_history(user_id)
            except Exception:
                raise exceptions.NotFound(f"User '{user_id}' was not found.")
        else:
            interactions = np.array(interactions)
            X = dataset.input_from_interactions(interactions)

        # prediction = model.predict(X, **predict_params)
        item_idxs = model.recommend_top_items(X, limit, **predict_params)
        items = dataset.indices_to_items(item_idxs)

        return json(items.to_dict("records"))

    @app.listener("after_server_stop")
    def on_shutdown(app, loop):
        logger.info("Server has been shut down.")

    return app


def run_server(port: int, models: Dict[Text, Model], dataset: Dataset) -> None:
    app = create_app(models, dataset)
    app.run(host="localhost", port=port, debug=False, access_log=False)
