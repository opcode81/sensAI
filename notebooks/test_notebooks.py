import json
import logging
import os
import pathlib
from glob import glob
from typing import Dict, Tuple

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_NOT_TESTED = [  # filenames of notebooks that are skipped in testing
    "intro_old.ipynb",
    "tracking_experiments.ipynb",
    "tensor_models_pytorch_lightning.ipynb",
    "clustering_evaluation.ipynb",
    "coordinate_clustering.ipynb",
    "temp.ipynb",
]  # TODO fix notebooks and remove these exclusions
log = logging.getLogger(__name__)


def notebooksUsedInDocs() -> Tuple[Dict[str, str], str]:
    for fname in os.listdir(DOCS_DIR):
        if "notebooks" in fname:
            path = DOCS_DIR / fname
            if os.path.isdir(path):
                with open(path / "notebooks_to_copy.json") as f:
                    notebooks_to_copy = json.load(f)
                for notebook_filename in notebooks_to_copy:
                    if not os.path.exists(NOTEBOOKS_DIR / notebook_filename):
                        raise FileNotFoundError(f"Notebook {notebook_filename} does not exist in notebooks directory")
                return notebooks_to_copy, str(path.absolute())
    raise Exception("Could not find notebooks directory in docs")


NOTEBOOKS_TO_COPY, DOCS_NOTEBOOKS_DIR = notebooksUsedInDocs()


class LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, notebookName, **kw):
        self._notebookName = notebookName
        super().__init__(**kw)

    def preprocess_cell(self, cell, resources, index):
        log.info(f"Processing cell {index} of {self._notebookName}")
        return super().preprocess_cell(cell, resources, index)


def execute_notebook(notebook_path):
    """
    Execute a jupyter notebook and return the executed notebook

    :param notebook_path: path to the notebook
    :return: the notebook with output cells added
    """
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = LoggingExecutePreprocessor(notebook_path, timeout=600)
    ep.preprocess(nb, resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    return nb


@pytest.mark.parametrize(
    "notebook", [file for file in os.listdir(NOTEBOOKS_DIR) if file.endswith(".ipynb") and file not in NOTEBOOKS_NOT_TESTED]
)
def test_notebook(notebook):
    """
    Tests notebooks in the /notebooks folder, optionally copying the executed notebook to the /docs folder

    :param notebook:
    :return:
    """
    notebook_path = NOTEBOOKS_DIR / notebook
    nb = execute_notebook(notebook_path)

    # saving the executed notebook to docs
    if notebook in NOTEBOOKS_TO_COPY:
        output_path = os.path.join(DOCS_NOTEBOOKS_DIR, NOTEBOOKS_TO_COPY[notebook])
        log.info(f"Saving executed notebook to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    else:
        log.info(f"Notebook {notebook} is not used in docs; not copied")


@pytest.mark.parametrize(
    "notebook_path", [p for p in [os.path.abspath(g) for g in glob(f"{DOCS_DIR}/**/*.ipynb")] if not p.startswith(DOCS_NOTEBOOKS_DIR)]
)
def test_build_docs_notebook_non_tutorial(notebook_path):
    """
    Process all notebooks in the docs/ folder that are purely documentation, i.e. ones that aren't tutorials which
    have been copied to DOCS_NOTEBOOKS_DIR by the other test above.

    Runs each notebook and saves it with output cells in-place.

    :param notebook_path:
    :return:
    """
    nb = execute_notebook(notebook_path)
    with open(notebook_path, "w", encoding="utf-8") as f:
        log.info(f"Saving executed notebook with outputs to {notebook_path}")
        nbformat.write(nb, f)
