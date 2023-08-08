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


def _get_from_endpoint(endpoint: str) -> list[dict]:
    try:
        token = _get_auth_token()
    except ValueError as e:
        logger.warn("Issue while loading from endpoints %s data:",
                    endpoint, exc_info=e)
        return {}
    headers = {
        "Authorization": f"Token {token}",
    }

    logger.debug("Querying MCT endpoint: %s", endpoint)

    django_api_url = f"{MCT_BASE_URL}{endpoint}"

    response = requests.get(django_api_url, headers=headers)

    j_dict = response.json()
    return j_dict['results']


def get_mct_project_data() -> list[dict]:
    response_data = _get_from_endpoint("project-annotate-entities/")

    output = response_data
    return output
