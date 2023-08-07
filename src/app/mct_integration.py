import json
import requests
import logging

from functools import cache


from .envs import MCT_BASE_URL, MCT_USERNAME, MCT_PASSWORD

logger = logging.getLogger(__name__)


@cache
def _get_auth_token(username: str = MCT_USERNAME,
                    password: str = MCT_PASSWORD) -> str:
    payload = {"username": username, "password": password}
    url = f"{MCT_BASE_URL}api-token-auth/"
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise ValueError(f"FAILED auth: {resp.status_code}")
    return json.loads(resp.text)["token"]


def get_mct_data() -> dict:
    try:
        token = _get_auth_token()
    except ValueError as e:
        logger.warn("Issue while loading MCT data:", exc_info=e)
        return {}
    headers = {
        "Authorization": f"Token {token}",
    }

    endpoint = "project-annotate-entities/"
    django_api_url = f"{MCT_BASE_URL}{endpoint}"

    response = requests.get(django_api_url, headers=headers)
    response_data = response.json()

    return response_data["results"]
