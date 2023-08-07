import logging
import os
import sys

import pytest

from model_test_case import IrisDataSet, ClassificationTestCase, RegressionTestCase, DiabetesDataSet, RESOURCE_DIR

sys.path.append(os.path.abspath("."))

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def testResources():
    return RESOURCE_DIR


@pytest.fixture(scope="session")
def irisDataSet():
    return IrisDataSet()


@pytest.fixture(scope="session")
def irisClassificationTestCase(irisDataSet):
    return ClassificationTestCase(irisDataSet.getInputOutputData())


@pytest.fixture(scope="session")
def diabetesDataSet():
    return DiabetesDataSet()


@pytest.fixture(scope="session")
def diabetesRegressionTestCase(diabetesDataSet):
    return RegressionTestCase(diabetesDataSet.getInputOutputData())
