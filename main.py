import asyncio
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import aiohttp
import anyio
import pymorphy2
from async_timeout import timeout
import pytest

from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize
from text_tools import calculate_jaundice_rate, split_by_words

logger = logging.getLogger(__name__)


@contextmanager
def catchtime():
    t1 = t2 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()


CHARGED_FILEPATHES = [
    "charged_dict/negative_words.txt",
    "charged_dict/positive_words.txt",
]
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


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ArticleResult):
            return {
                "url": obj.url,
                "status": obj.status.value,
                "words_count": obj.words_count,
                "score": obj.score,
            }
        return json.JSONEncoder.default(self, obj)


RESPONSE_TIMEOUT = 1
MORPH_TIMEOUT = 3


async def process_article(session, morph, charged_words, url, results):
    try:
        async with timeout(RESPONSE_TIMEOUT):
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


@pytest.mark.asyncio
async def test_process_article():
    charged_words = load_charged_words(CHARGED_FILEPATHES)
    morph = pymorphy2.MorphAnalyzer()
    async with aiohttp.ClientSession() as session:
        # non inosmi site
        url = "https://www.google.com"
        results = []
        await process_article(session, morph, charged_words, url, results)
        assert results[0].status == ProcessingStatus.PARSING_ERROR
        # non existing url
        url = "https://inosmi.ru/20251126/su-57-275814.html"
        results = []
        await process_article(session, morph, charged_words, url, results)
        assert results[0].status == ProcessingStatus.FETCH_ERROR
        # existing url
        url = "https://inosmi.ru/20251126/polsha-275819902.html"
        results = []
        await process_article(session, morph, charged_words, url, results)
        assert results[0].status == ProcessingStatus.OK
        # Timeout request
        global RESPONSE_TIMEOUT
        RESPONSE_TIMEOUT = 0.00001
        url = "https://inosmi.ru/20251126/polsha-275819902.html"
        results = []
        await process_article(session, morph, charged_words, url, results)
        assert results[0].status == ProcessingStatus.TIMEOUT
        # Timeout morph
        global MORPH_TIMEOUT
        MORPH_TIMEOUT = 0.001
        url = "https://inosmi.ru/20251126/polsha-275819902.html"
        results = []
        await process_article(session, morph, charged_words, url, results)
        assert results[0].status == ProcessingStatus.TIMEOUT


async def fetch(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


def load_charged_words(filepathes=CHARGED_FILEPATHES):
    chardged_words = []
    for filepath in filepathes:
        with open(filepath, "r", encoding="utf-8") as f:
            chardged_words.extend(f.read().splitlines())
    return chardged_words


async def main(urls):
    charged_words = load_charged_words(CHARGED_FILEPATHES)
    morph = pymorphy2.MorphAnalyzer()
    results = []
    async with aiohttp.ClientSession() as session:
        async with anyio.create_task_group() as tg:
            for url in urls:
                tg.start_soon(
                    process_article, session, morph, charged_words, url, results
                )
    logger.info(results)
    return results


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    asyncio.run(main(TEST_ARTICLES))
