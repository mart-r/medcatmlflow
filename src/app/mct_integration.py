import json

import requests

from functools import cache


from .envs import MCT_BASE_URL, MCT_USERNAME, MCT_PASSWORD


@cache
def _get_auth_token(username: str = MCT_USERNAME,
                    password: str = MCT_PASSWORD) -> str:
    payload = {"username": username, "password": password}
    url = f"{MCT_BASE_URL}api-token-auth/"
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        return f"FAILED: {resp.status_code}"
    return json.loads(resp.text)["token"]


def get_mct_data() -> dict:
    token = _get_auth_token()
    headers = {
        "Authorization": f"Token {token}",
    }

    endpoint = "project-annotate-entities/"
    django_api_url = f"{MCT_BASE_URL}{endpoint}"

    response = requests.get(django_api_url, headers=headers)
    response_data = response.json()

    return response_data["results"]
