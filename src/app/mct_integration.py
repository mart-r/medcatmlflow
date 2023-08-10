from typing import Optional
import json
import requests
import logging
import tempfile
from functools import cache
import re
from urllib.parse import urlparse

from .medcat_integration import get_cdb_hash
from .mlflow_integration import register_mct_cdbs
from .utils import expire_cache_after
from .envs import MCT_BASE_URL, MCT_USERNAME, MCT_PASSWORD

logger = logging.getLogger(__name__)


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
    correct_port = PORT_PATTERN.search(MCT_BASE_URL).group(1)
    current_port_match = PORT_PATTERN.search(cdb_file_url)
    if current_port_match:
        current_port = current_port_match.group(1)
        url_fixed_port = cdb_file_url.replace(current_port, correct_port)
    else:  # no port in URL
        protocol_and_ip, endpoint = split_url(cdb_file_url)
        url_fixed_port = f"{protocol_and_ip}{correct_port}{endpoint}"
    logger.info("Fixed port from '%s' to '%s", cdb_file_url, url_fixed_port)
    return _download_url(url_fixed_port)


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


def _get_from_endpoint(endpoint: str) -> list[dict]:
    try:
        headers = _get_token_header()
    except ValueError as e:
        logger.warn("Issue while loading from endpoints %s data:",
                    endpoint, exc_info=e)
        return []

    logger.debug("Querying MCT endpoint: %s", endpoint)

    django_api_url = f"{MCT_BASE_URL}{endpoint}"

    response = requests.get(django_api_url, headers=headers)

    j_dict = response.json()
    return j_dict["results"]


@expire_cache_after(60)  # expire every minute
def _get_all_cdbs() -> list[dict]:
    return _get_from_endpoint("concept-dbs/")


@expire_cache_after(60)  # expire every minute
def _get_cdb(cdb_id) -> tuple[str, Optional[str]]:
    cdbs = _get_all_cdbs()
    # e.g:
    # [
    #   {'id': 5, 'name': 'snomed_cdb',
    #    'cdb_file': 'http://10.211.114.213/media/snomed-cdb.dat',
    #    'use_for_training': True},
    #   {'id': 6, 'name': 'umls_cdb',
    #    'cdb_file': 'http://10.211.114.213/media/cdb-medmen-v1.dat',
    #    'use_for_training': True}
    # ]
    for saved_cdb in cdbs:
        cur_id = saved_cdb["id"]
        if cdb_id == cur_id:
            cdb_file = saved_cdb["cdb_file"]
            return f"{saved_cdb['name']} (ID: {cdb_id})", cdb_file
    return "Unknown", None


@expire_cache_after(60)  # expire every minute
def _get_all_datasets() -> list[dict]:
    return _get_from_endpoint("datasets/")


@expire_cache_after(60)  # expire every minute
def _get_dataset(dataset_id) -> str:
    datasets = _get_all_datasets()
    # e.g:
    # [
    #   {'id': 2, 'name': 'Example Dataset',
    #    'original_file': 'http://10.211.114.213/media/Example_Dataset.csv',
    #    'create_time': '2023-06-14T22:46:38.746998Z',
    #    'description': 'Example clinical text ...'},
    #   {'id': 3, 'name': 'Example Dataset',
    #    'original_file': 'http://10.211.114.213/media/Example_Dataset_cp.csv',
    #    'create_time': '2023-06-14T22:47:08.817644Z',
    #    'description': 'Example clinical text ...'}
    # ]
    for saved_ds in datasets:
        cur_id = saved_ds["id"]
        if dataset_id == cur_id:
            return f"{saved_ds['name']} (ID: {dataset_id})"
    return "Unknown"


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


def get_mct_project_data() -> list[dict]:
    response_data = _get_from_endpoint("project-annotate-entities/")

    output = []
    used_cdb_ids = set()
    for project in response_data:
        name = project["name"]
        cdb_id = project["concept_db"]
        if cdb_id in used_cdb_ids:
            continue  # using CDB that was already handled
        used_cdb_ids.add(cdb_id)
        dataset_id = project["dataset"]
        cdb_descr, cdb_file = _get_cdb(cdb_id)
        logger.info("CDB (ID, descr, file): %s, %s, %s",
                    cdb_id, cdb_descr, cdb_file)
        output.append(
            {"name": name,
             "cdb": cdb_descr,
             "dataset": _get_dataset(dataset_id),
             "cdb_ID": cdb_id,
             "cdb_HASH": _get_hash_for_cdb(cdb_id, cdb_file)
             }
        )
    logger.info("Found %d MCT project data", len(output))
    # populate MCT hashes
    for cdb in output:
        register_mct_cdbs(cdb["cdb_ID"], cdb["cdb"], cdb["cdb_HASH"])
    return output
