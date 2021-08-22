import logging
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "docs"
resources = {"metadata": {"path": NOTEBOOKS_DIR}}

log = logging.getLogger(__name__)


class LoggingExecutePreprocessor(ExecutePreprocessor):
    def __init__(self, notebookName, **kw):
        self._notebookName = notebookName
        super().__init__(**kw)

    def preprocess_cell(self, cell, resources, index):
        log.info(f"Processing cell {index} of {self._notebookName}")
        return super().preprocess_cell(cell, resources, index)


@pytest.mark.parametrize(
    "notebook", [file for file in os.listdir(NOTEBOOKS_DIR) if file.endswith(".ipynb")]
)
def test_notebook(notebook):
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook)
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = LoggingExecutePreprocessor(notebook, timeout=600)
    ep.preprocess(nb, resources=resources)

    # saving the executed notebook to docs
    output_path = os.path.join(DOCS_DIR, notebook)
    log.info(f"Saving executed notebook to {output_path} for documentation purposes")
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
