# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
from itertools import islice

import requests
from fastapi import HTTPException
from loguru import logger

# Search engine related. You don't really need to change this.
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SEARCHAPI_SEARCH_ENDPOINT = "https://www.searchapi.io/api/v1/search"
# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5


def search_with_bing(query: str, subscription_key: str):
    """
    Search with bing and return the contexts.
    """
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_google(query: str, subscription_key: str, cx: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_serper(query: str, subscription_key: str):
    """
    Search with serper and return the contexts.
    """
    payload = json.dumps({
        "q": query,
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    })
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content["knowledgeGraph"].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append({
                    "name": json_content["knowledgeGraph"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content["answerBox"].get("answer")
            if url and snippet:
                contexts.append({
                    "name": json_content["answerBox"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []


def search_with_searchapi(query: str, subscription_key: str):
    """
    Search with SearchApi.io and return the contexts.
    """
    payload = {
        "q": query,
        "engine": "google",
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    }
    headers = {"Authorization": f"Bearer {subscription_key}", "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SEARCHAPI_SEARCH_ENDPOINT}"
    )
    response = requests.get(
        SEARCHAPI_SEARCH_ENDPOINT,
        headers=headers,
        params=payload,
        timeout=30,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []

        if json_content.get("answer_box"):
            if json_content["answer_box"].get("organic_result"):
                title = json_content["answer_box"].get("organic_result").get("title", "")
                url = json_content["answer_box"].get("organic_result").get("link", "")
            if json_content["answer_box"].get("type") == "population_graph":
                title = json_content["answer_box"].get("place", "")
                url = json_content["answer_box"].get("explore_more_link", "")

            title = json_content["answer_box"].get("title", "")
            url = json_content["answer_box"].get("link")
            snippet = json_content["answer_box"].get("answer") or json_content["answer_box"].get("snippet")

            if url and snippet:
                contexts.append({
                    "name": title,
                    "url": url,
                    "snippet": snippet
                })

        if json_content.get("knowledge_graph"):
            if json_content["knowledge_graph"].get("source"):
                url = json_content["knowledge_graph"].get("source").get("link", "")

            url = json_content["knowledge_graph"].get("website", "")
            snippet = json_content["knowledge_graph"].get("description")

            if url and snippet:
                contexts.append({
                    "name": json_content["knowledge_graph"].get("title", ""),
                    "url": url,
                    "snippet": snippet
                })

        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic_results"]
        ]

        if json_content.get("related_questions"):
            for question in json_content["related_questions"]:
                if question.get("source"):
                    url = question.get("source").get("link", "")
                else:
                    url = ""

                snippet = question.get("answer", "")

                if url and snippet:
                    contexts.append({
                        "name": question.get("question", ""),
                        "url": url,
                        "snippet": snippet
                    })

        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []


def search_with_duckduckgo(query: str):
    """
    Search with DuckDuckGo and return the contexts.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("Please install duckduckgo-search to use this search engine.")
    contexts = []
    with DDGS() as ddgs:
        ddgs_gen = ddgs.text(query, backend="lite")
        for r in islice(ddgs_gen, REFERENCE_COUNT):
            contexts.append({
                "name": r['title'],
                "url": r['href'],
                "snippet": r['body']
            })
    return contexts
