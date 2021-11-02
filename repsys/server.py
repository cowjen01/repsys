from sanic import Sanic
from sanic.response import json, file

app = Sanic(__name__)

static_folder = "../frontend/build"
app.static("/", static_folder)


@app.route("/")
async def index(request):
    return await file(f"{static_folder}/index.html")


@app.route("/api/models")
def get_models(request):
    return json(
        [
            {
                "key": "knn",
                "attributes": [
                    {
                        "key": "n",
                        "label": "Neighbors",
                        "type": "number",
                        "defaultValue": 5,
                    },
                ],
            },
            {
                "key": "vasp",
                "attributes": [
                    {
                        "key": "h",
                        "label": "Some parameter",
                        "type": "text",
                    },
                ],
                "businessRules": ["popularity", "explore"],
            },
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
            {
                "id": "6",
            },
            {
                "id": "7",
            },
            {
                "id": "8",
            },
            {
                "id": "9",
            },
            {
                "id": "10",
            },
            {
                "id": "11",
            },
            {
                "id": "12",
            },
            {
                "id": "13",
            },
            {
                "id": "14",
            },
            {
                "id": "15",
            },
            {
                "id": "16",
            },
            {
                "id": "17",
            },
            {
                "id": "18",
            },
            {
                "id": "19",
            },
            {
                "id": "20",
            },
        ]
    )


@app.listener("after_server_stop")
def on_shutdown(app, loop):
    print("I am done!")
