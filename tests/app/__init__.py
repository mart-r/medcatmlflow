import os

TESTS_APP_PATH = os.path.abspath(os.path.dirname(__file__))
TESTS_PATH = os.path.abspath(os.path.join(TESTS_APP_PATH, os.path.pardir))
TESTS_RESOURCES_PATH = os.path.join(TESTS_PATH, "resources")
