from typing import Optional, List
import json
import requests
import logging
import tempfile
from functools import lru_cache
import re
from urllib.parse import urlparse

from .medcat_integration import get_cdb_hash
from ..main.utils import expire_cache_after
from ..main.envs import MCT_BASE_URL, MCT_USERNAME, MCT_PASSWORD

logger = logging.getLogger(__name__)

cache = lru_cache(None)


# port with colon and slash on group 1
PORT_PATTERN = re.compile(r"http://[^:]+(:\d+/?)")


def split_url(url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    path = parsed_url.path
    return base_url, path


def download_cdb(cdb_file_url: str) -> str:
    # the URL comes without the port
    # so I need to fix that
    if MCT_BASE_URL is None:
        raise ValueError("No MCT_BASE_URL defined "
                         "- cannot use MedCATtrainer stuff")
    matched = PORT_PATTERN.search(MCT_BASE_URL)
    if matched:
        correct_port = matched.group(1)
    else:
        correct_port = "80"  # DEFAULT to 80
        logger.warning("No port found in MCT base URL (%s) - using %s instead",
                       MCT_BASE_URL, correct_port)
    current_port_match = PORT_PATTERN.search(cdb_file_url)
    if current_port_match:
        current_port = current_port_match.group(1)
        url_fixed_port = cdb_file_url.replace(current_port, correct_port)
    else:  # no port in URL
        protocol_and_ip, endpoint = split_url(cdb_file_url)
        url_fixed_port = f"{protocol_and_ip}{correct_port}{endpoint}"
    logger.info("Fixed port from '%s' to '%s", cdb_file_url, url_fixed_port)
    saved_file_name = _download_url(url_fixed_port)
    if not saved_file_name:
        raise ValueError(f"Unable to find CDB from {cdb_file_url}")
    return saved_file_name


def _download_url(url: str) -> Optional[str]:
    # Send a GET request to the URL with headers
    try:
        headers = _get_token_header()
    except ValueError as e:
        logger.warn("Issue while downloading from '%s':", url, exc_info=e)
        return None
    response = requests.get(url, headers=headers, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        file_extension = url.split(".")[-1]
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}", delete=False
        ) as f:
            # Iterate over the response content in chunks and write to the file
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("File '%s' downloaded successfully.", f.name)
        return f.name
    else:
        code = response.status_code
        logger.warning("Failed to download the file. Status code: %s", code)
        return None


@expire_cache_after(10 * 60)  # expire every 10 minutes
def _get_auth_token(username: str = MCT_USERNAME,
                    password: str = MCT_PASSWORD) -> str:
    logger.info("Getting new authentication token")
    payload = {"username": username, "password": password}
    url = f"{MCT_BASE_URL}api-token-auth/"
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise ValueError(f"FAILED auth: {resp.status_code}")
    return json.loads(resp.text)["token"]


def _get_token_header() -> dict:
    token = _get_auth_token()
    return {
        "Authorization": f"Token {token}",
    }


def _get_from_endpoint(endpoint: str) -> List[dict]:
    try:
        headers = _get_token_header()
    except ValueError as e:
        logger.warn("Issue while loading from endpoints %s data:",
                    endpoint, exc_info=e)
        return []
    except requests.exceptions.ConnectionError as e:
        logger.error("Issue getting token - "
                     "did you set up MedCATtrainer URL "
                     "(MCT_BASE_URL) correctly?", exc_info=e)
        return []

    logger.debug("Querying MCT endpoint: %s", endpoint)

    django_api_url = f"{MCT_BASE_URL}{endpoint}"

    response = requests.get(django_api_url, headers=headers)

    j_dict = response.json()
    return j_dict["results"]


@expire_cache_after(60)  # expire every minute
def _get_all_cdbs() -> List[dict]:
    return _get_from_endpoint("concept-dbs/")


# TODO - instead of caching every time, save this somewhere
@cache
def _get_hash_for_cdb(cdb_id: str, cdb_file: str) -> Optional[str]:
    logger.debug("Getting hash for CDB '%s' (%s)", cdb_id, cdb_file)
    temp_file = download_cdb(cdb_file)
    if not temp_file:
        logger.error("Could not find CDB for ID '%s' at '%s'",
                     cdb_id, cdb_file)
        return None
    return get_cdb_hash(temp_file)


@expire_cache_after(60)
def get_mct_cdb_id(cdb_hash: str) -> Optional[str]:
    for cdb in _get_all_cdbs():
        cdb_id = cdb["id"]
        cdb_file = cdb["cdb_file"]
        try:
            cur_hash = _get_hash_for_cdb(cdb_id, cdb_file)
        except Exception as e:
            logger.warning("Unable to get MCT CDB hash for cdb '%s'",
                           cdb["id"], exc_info=e)
            continue
        if cur_hash == cdb_hash:
            return cdb_id
    return None
