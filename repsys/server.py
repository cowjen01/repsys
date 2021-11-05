import logging
import os

from sanic import Sanic
from sanic.response import json, file

logger = logging.getLogger(__name__)


def create_app(model_loader):
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
                        for p in model_loader.instances[k].prediction_params()
                    ],
                }
                for k in model_loader.instances.keys()
            ]
        )

    @app.route("/api/users")
    def get_users(request):
        return json(
            [
                {
                    "id": "1",
                },
                {
                    "id": "2",
                },
                {
                    "id": "3",
                },
                {
                    "id": "4",
                },
                {
                    "id": "5",
                },
            ]
        )

    @app.listener("after_server_stop")
    def on_shutdown(app, loop):
        logger.info("I am done!")

    return app


def run_server(port, model_loader) -> None:
    app = create_app(model_loader)
    app.run(host="localhost", port=port)
