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
    return j_dict["results"]


@cache  # TODO - limit caching? shouldn't be an issue with small deployments
def _get_cdb(cdb_id) -> str:
    cdbs = _get_from_endpoint("concept-dbs/")
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
        cur_id = saved_cdb['id']
        if cdb_id == cur_id:
            return f"{saved_cdb['name']} (ID: {cdb_id})"
    return "Unknown"


@cache  # TODO - limit caching? shouldn't be an issue with small deployments
def _get_dataset(dataset_id) -> str:
    datasets = _get_from_endpoint("datasets/")
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
        cur_id = saved_ds['id']
        if dataset_id == cur_id:
            return f"{saved_ds['name']} (ID: {dataset_id})"
    return "Unknown"


def get_mct_project_data() -> list[dict]:
    response_data = _get_from_endpoint("project-annotate-entities/")

    output = []
    for project in response_data:
        name = project["name"]
        cdb_id = project["concept_db"]
        dataset_id = project["dataset"]
        output.append(
            {"name": name,
             "cdb": _get_cdb(cdb_id),
             "dataset": _get_dataset(dataset_id)
             }
        )
    logger.info("Found %d MCT project data", len(output))
    return output
