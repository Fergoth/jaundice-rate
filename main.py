import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import aiohttp
import logging
import anyio
import pymorphy2
from async_timeout import timeout

from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words

logger = logging.getLogger(__name__)


@contextmanager
def catchtime():
    t1 = t2 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()


TEST_ARTICLES = [
    "https://inosmi.ru/20251126/su-57-275814.html",
    "https://lenta.ru/brief/2021/08/26/afg_terror/",
    "https://inosmi.ru/20251126/ermak-275824488.html",
    "https://inosmi.ru/20251126/mirnyy_plan-275824175.html",
    "https://inosmi.ru/20251126/polsha-275819902.html",
]


class ProcessingStatus(Enum):
    OK = "OK"
    FETCH_ERROR = "FETCH_ERROR"
    PARSING_ERROR = "PARSING_ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass
class ArticleResult:
    url: str
    status: ProcessingStatus
    words_count: int | None = None
    score: float | None = None

    def __repr__(self):
        return f"""
        URL: {self.url}
        Статус : {self.status}
        Слов в статье: {self.words_count}
        Рейтинг: {self.score}
    """


RESPONSE_TIMEOUT = 1
MORPH_TIMEOUT = 3


async def process_article(session, morph, charged_words, url, results):
    try:
        async with timeout(1):
            html = await fetch(session, url)
    except aiohttp.ClientError:
        results.append(ArticleResult(url, ProcessingStatus.FETCH_ERROR))
        return
    except asyncio.TimeoutError:
        results.append(ArticleResult(url, ProcessingStatus.TIMEOUT))
        return
    try:
        sanitized_text = sanitize(html, plaintext=True)
    except ArticleNotFound:
        results.append(ArticleResult(url, ProcessingStatus.PARSING_ERROR))
        return
    try:
        async with timeout(MORPH_TIMEOUT):
            with catchtime() as morph_time:
                words = await split_by_words(morph, sanitized_text)
    except asyncio.TimeoutError:
        results.append(ArticleResult(url, ProcessingStatus.TIMEOUT))
        return
    logger.info(f"Анализ закончен за {morph_time():.5f} сек")
    score = calculate_jaundice_rate(words, charged_words=charged_words)
    results.append(ArticleResult(url, ProcessingStatus.OK, len(words), score))


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


def load_charged_words(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().splitlines()


async def main():
    charged_words = load_charged_words("charged_dict/negative_words.txt")
    charged_words.extend(load_charged_words("charged_dict/positive_words.txt"))
    morph = pymorphy2.MorphAnalyzer()
    results = []
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in TEST_ARTICLES:
                tg.start_soon(
                    process_article, session, morph, charged_words, url, results
                )
    print(results)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    asyncio.run(main())
