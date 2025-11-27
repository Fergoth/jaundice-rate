import json
from functools import partial

from aiohttp import ClientSession, web
from anyio import create_task_group
from pymorphy2 import MorphAnalyzer

from main import MyJsonEncoder, load_charged_words, process_article

MAX_URLS = 5


async def handle(request, morph, charged_words):
    param = request.rel_url.query["urls"]
    urls = param.split(",")
    if len(urls) > MAX_URLS:
        raise web.HTTPBadRequest(
            text=f"too many urls in request, should be {MAX_URLS} or less"
        )
    results = []
    async with ClientSession() as session:
        async with create_task_group() as tg:
            for url in urls:
                tg.start_soon(
                    process_article, session, morph, charged_words, url, results
                )
    my_json_dumps = partial(json.dumps, cls=MyJsonEncoder)
    return web.json_response(results, dumps=my_json_dumps)


if __name__ == "__main__":
    charged_words = load_charged_words()
    morph = MorphAnalyzer()
    handle = partial(handle, morph=morph, charged_words=charged_words)
    app = web.Application()
    app.add_routes([web.get("/", handle)])
    web.run_app(app)
