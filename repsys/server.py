import logging
import os
from typing import Dict

import numpy as np
from pandas import DataFrame
from sanic import Sanic
from sanic.exceptions import InvalidUsage, NotFound
from sanic.response import json, file

import repsys.dtypes as dtypes
from repsys.dataset import Dataset
from repsys.dtypes import filter_columns_by_type
from repsys.model import Model

logger = logging.getLogger(__name__)


def create_app(models: Dict[str, Model], dataset: Dataset):
    app = Sanic(__name__)

    static_folder = os.path.join(os.path.dirname(__file__), "../frontend/build")
    app.static("/", static_folder)

    def serialize_items(items: DataFrame):
        items_copy = items.copy()
        items_copy["id"] = items_copy.index

        tag_cols = filter_columns_by_type(dataset.item_cols(), dtypes.Tag)
        for col in tag_cols:
            items_copy[col] = items_copy[col].str.join(', ')

        return items_copy.to_dict("records")

    def get_item_attributes() -> Dict[str, any]:
        attributes = {}
        for col, datatype in dataset.item_cols().items():
            attributes[col] = {'dtype': str(datatype)}

            if type(datatype) == dtypes.Tag:
                attributes[col]['options'] = dataset.tags.get(col)

            if type(datatype) == dtypes.Category:
                attributes[col]['options'] = dataset.categories.get(col)

            if type(datatype) == dtypes.Number:
                hist = dataset.histograms.get(col)
                attributes[col]['bins'] = hist[1].astype(int).tolist()

        return attributes

    @app.route("/")
    async def index(request):
        return await file(f"{static_folder}/index.html")

    @app.route("/api/models")
    def get_config(request):
        return json({
            model.name(): model.to_dict() for model in models.values()
        })

    @app.route("/api/dataset")
    def get_config(request):
        return json({
            "totalItems": dataset.get_total_items(),
            "attributes": get_item_attributes()
        })

    @app.route("/api/users")
    def get_users(request):
        split = request.args.get("split")

        if not split:
            raise InvalidUsage("The dataset's split must be specified.")

        if split not in ['train', 'validation', 'test']:
            raise InvalidUsage("The split must be one of: train, validation or test.")

        users = dataset.get_users_by_split(split)
        return json(users)

    @app.route("/api/items")
    def get_items(request):
        query = request.args.get("query")

        if not query:
            raise InvalidUsage("The query string must be specified.")

        if len(query) < 3:
            raise InvalidUsage("The query must have at least 3 characters.")

        items = dataset.get_items_by_title(query)
        data = json(serialize_items(items))

        return data

    @app.route("/api/users/<uid>")
    def get_interactions(request, uid: str):
        split = dataset.get_split_by_user(uid)

        if not split:
            raise NotFound(f"User '{uid}' not found.")

        items = dataset.get_interacted_items_by_user(uid, split.name)
        data = json({
            'interactions': serialize_items(items)
        })

        return data

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
            raise InvalidUsage(
                "Either the user or his interactions must be specified."
            )

        if not model_name:
            raise InvalidUsage("Model name must be specified.")

        model = [m for m in models if m.name() == model_name][0]

        if not model:
            raise NotFound(f"Model '{model_name}' was not found.")

        default_params = {p.name: p.default for p in model.web_params()}
        cleaned_params = {
            k: v for k, v in params.items() if k in default_params
        }
        predict_params = {**default_params, **cleaned_params}

        if user_id is not None:
            try:
                user_id = int(user_id)
                X = dataset.get_user_history(user_id)
            except Exception:
                raise NotFound(f"User '{user_id}' was not found.")
        else:
            interactions = np.array(interactions)
            X = dataset.input_from_interactions(interactions)

        items = model.predict_top_n(X, limit, **predict_params)
        data = json(serialize_items(items))

        return data

    @app.listener("after_server_stop")
    def on_shutdown(current_app, loop):
        logger.info("Server has been shut down.")

    return app


def run_server(port: int, models: Dict[str, Model], dataset: Dataset) -> None:
    app = create_app(models, dataset)
    app.config.FALLBACK_ERROR_FORMAT = "json"
    app.run(host="localhost", port=port, debug=False, access_log=False)
