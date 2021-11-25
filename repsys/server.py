import logging
import os
from sanic import Sanic
from sanic.response import json, file
import numpy as np
from sanic import exceptions
from scipy import sparse

from repsys.core import RepsysCore

logger = logging.getLogger(__name__)


def create_app(core: RepsysCore):
    app = Sanic(__name__)

    static_folder = os.path.join(os.path.dirname(__file__), "../frontend/build")
    app.static("/", static_folder)

    @app.route("/")
    async def index(request):
        return await file(f"{static_folder}/index.html")

    @app.route("/api/models")
    def get_models(request):
        return json([m.to_dict() for m in core.models.keys()])

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

        model = core.get_model(model_name)

        if not model:
            raise exceptions.NotFound(f"Model '{model_name}' was not found.")

        default_params = {
            p.key: p.default_value for p in model.website_params()
        }
        cleaned_params = {
            k: v for k, v in params.items() if k in default_params
        }
        predict_params = {**default_params, **cleaned_params}

        if user_index is not None:
            try:
                user_index = int(user_index)
                X = core.get_user_history(user_index)
            except Exception:
                raise exceptions.NotFound(f"User '{user_index}' was not found.")

        else:
            interactions = np.array(interactions)
            X = core.from_interactions(interactions)

        prediction = model.predict(X, **predict_params)
        items = core.prediction_to_items(prediction, limit)

        return json(items.to_dict("records"))

    @app.route("/api/users")
    def get_users(request):
        return json(core.dataset.vad_users.to_dict("records"))

    @app.route("/api/items")
    def get_items(request):
        query_str = request.args.get("query")

        if not query_str or len(query_str) == 0:
            return json([])

        items = core.filter_items("title", query_str)

        return json(items.to_dict("records"))

    @app.route("/api/interactions")
    def get_interactions(request):
        user_index = request.args.get("user")

        if user_index is None:
            raise exceptions.InvalidUsage("User must be specified.")

        try:
            user_index = int(user_index)
            items = core.get_interacted_items(user_index)
        except Exception:
            raise exceptions.NotFound(f"User '{user_index}' was not found.")

        return json(items.to_dict("records"))

    @app.listener("after_server_stop")
    def on_shutdown(app, loop):
        logger.info("Calling models to save their state.")

        core.save_models()

        logger.info("Server has been shut down.")

    return app


def run_server(port: int, core: RepsysCore) -> None:
    app = create_app(core)
    app.run(host="localhost", port=port, debug=False, access_log=False)
