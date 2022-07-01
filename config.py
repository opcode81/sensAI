import json
import logging.handlers
import os

from typing import List, Union, Dict


log = logging.getLogger(__name__)

__config_instance = None

topLevelDirectory = os.path.abspath(os.path.dirname(__file__))


class __Configuration:
    """
    Holds essential configuration entries
    """
    log = log.getChild(__qualname__)

    PROCESSED = "processed"
    RAW = "raw"
    CLEANED = "cleaned"
    GROUND_TRUTH = "ground_truth"
    DATA = "data"

    def __init__(self, config_files: List[str] = None):
        """
        :param config_files: list of JSON configuration files (relative to root) from which to read.
            If None, reads from './config.json' and './config_local.json' (latter files have precedence)
        """
        if config_files is None:
            config_files = ["config.json", "config_local.json"]
        self.config = {}
        for filename in config_files:
            file_path = os.path.join(topLevelDirectory, filename)
            if os.path.exists(file_path):
                self.log.info("Reading configuration from %s" % file_path)
                with open(file_path, 'r') as f:
                    self.config.update(json.load(f))
        if not self.config:
            raise Exception("No configuration entries could be read from %s" % config_files)

    def _get_non_empty_entry(self, key: Union[str, List[str]]) -> Union[float, str, List, Dict]:
        """
        Retrieves an entry from the configuration

        :param key: key or list of keys to go through hierarchically
        :return: the queried json object
        """
        if isinstance(key, str):
            key = [key]
        value = self.config
        for k in key:
            value = value.get(k)
            if value is None:
                raise Exception(f"Value for key '{key}' not set in configuration")
        return value

    def _get_path(self, key: Union[str, List[str]], create=False) -> str:
        """
        Retrieves an existing local path from the configuration

        :param key: key or list of keys to go through hierarchically
        :param create: if True, a directory with the given path will be created on the fly.
        :return: the queried path
        """
        path_string = self._get_non_empty_entry(key)
        path = os.path.abspath(os.path.join(topLevelDirectory, path_string))
        if not os.path.exists(path):
            if isinstance(key, list):
                key = ".".join(key)  # purely for logging
            if create:
                log.info(f"Configured directory {key}='{path}' not found; will create it")
                os.makedirs(path)
            else:
                raise FileNotFoundError(f"Configured directory {key}='{path}' does not exist.")
        return path.replace("/", os.sep)

    @property
    def artifacts(self):
        return self._get_path("artifacts", create=True)

    @property
    def visualizations(self):
        return self._get_path("visualizations", create=True)

    @property
    def temp(self):
        return self._get_path("temp", create=True)

    @property
    def data(self):
        return self._get_path("data")

    @property
    def data_raw(self):
        return self._get_path("data_raw")

    @property
    def data_cleaned(self):
        return self._get_path("data_cleaned", create=True)

    @property
    def data_processed(self):
        return self._get_path("data_processed", create=True)

    @property
    def data_ground_truth(self):
        return self._get_path("data_ground_truth")

    def datafile_path(self, filename: str, stage="raw"):
        """
        Retrieve absolute path to an existing datafile
        :param filename:
        :param stage: raw, ground_truth, cleaned or processed
        :return: path
        """
        if stage == self.RAW:
            basedir = self.data_raw
        elif stage == self.CLEANED:
            basedir = self.data_cleaned
        elif stage == self.PROCESSED:
            basedir = self.data_processed
        elif stage == self.GROUND_TRUTH:
            basedir = self.data_ground_truth
        else:
            raise KeyError(f"Unknown stage: {stage}")
        full_path = os.path.join(basedir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No such file: {full_path}")
        return full_path

    def artifact_path(self, name: str):
        """
        Retrieve absolute path to an existing artifact
        :param name:
        :return:
        """
        full_path = os.path.join(self.artifacts, name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No such file: {full_path}")
        return full_path


def get_config(reload=False) -> __Configuration:
    """
    :param reload: if True, the configuration will be reloaded from the json files
    :return: the configuration instance
    """
    global __config_instance
    if __config_instance is None or reload:
        __config_instance = __Configuration()
    return __config_instance
