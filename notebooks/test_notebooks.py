import logging
import os
import pathlib
import re

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
NOTEBOOKS_NOT_TESTED = [
    "intro_old.ipynb",
    "tracking_experiments.ipynb",
    "tensor_models_pytorch_lightning.ipynb",
    "clustering_evaluation.ipynb",
    "coordinate_clustering.ipynb",
]  # TODO fix notebooks and remove these exclusions
log = logging.getLogger(__name__)


def notebooksUsedInDocs():
    with open(ROOT_DIR / "docs/index.rst", "r") as f:
        content = f.read()
    return re.findall(r"\s(\w+\.ipynb)", content)


NOTEBOOKS_TO_COPY = notebooksUsedInDocs()


class LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, notebookName, **kw):
        self._notebookName = notebookName
        super().__init__(**kw)

    def preprocess_cell(self, cell, resources, index):
        log.info(f"Processing cell {index} of {self._notebookName}")
        return super().preprocess_cell(cell, resources, index)


@pytest.mark.parametrize(
    "notebook", [file for file in os.listdir(NOTEBOOKS_DIR) if file.endswith(".ipynb") and file not in NOTEBOOKS_NOT_TESTED]
)
def test_notebook(notebook):
    notebook_path = NOTEBOOKS_DIR / notebook
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = LoggingExecutePreprocessor(notebook, timeout=600)
    ep.preprocess(nb, resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})

    # saving the executed notebook to docs
    if notebook in NOTEBOOKS_TO_COPY:
        output_path = os.path.join(DOCS_DIR, notebook)
        log.info(f"Saving executed notebook to {output_path} for documentation purposes")
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    else:
        log.info(f"Notebook {notebook} is not used in docs; not copied")