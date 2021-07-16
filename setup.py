import re
from typing import Iterable, Dict

from setuptools import setup, find_namespace_packages

tf_requirements = ['tensorflow==1.15.0']
torch_requirements = ['torch==1.4.0', 'torchtext==0.5.0']
lightgbm_requirements = ['lightgbm==2.3.0']
geoanalytics_requirements = ['networkx==2.4', 'Shapely~=1.7.0', 'geopandas==0.7.0']


# list of dependencies where ==/~= dependencies (used by us, particularly in requirements.txt) are relaxed:
# any later version is OK (as long as we are not aware of a concrete limitation - and once we are, we shall define
# the respective upper bound below)
DEPS_VERSION_LOWER_BOUND = ["pandas", "scipy", "numpy", "scikit-learn", "seaborn", "typing-extensions"]
DEPS_VERSION_UPPER_BOUND_EXCLUSIVE: Dict[str, str] = {}


def required_packages(deps: Iterable[str]):
    updated_deps = []
    for dep in deps:
        dep = dep.strip()
        m = re.match(r'([\w-]+)[=~]=', dep)  # match package with == or ~= version spec
        if m:
            package = m.group(1)
            if package in DEPS_VERSION_LOWER_BOUND:
                dep = dep.replace("==", ">=").replace("~=", ">=")
            elif package in DEPS_VERSION_UPPER_BOUND_EXCLUSIVE:
                dep = dep.replace("==", ">=").replace("~=", ">=")
                dep += ",<" + DEPS_VERSION_UPPER_BOUND_EXCLUSIVE[package]
        updated_deps.append(dep)
    return updated_deps


setup(
    name='sensai',
    package_dir={"": "src"},
    license="MIT",
    url="https://github.com/jambit/sensAI",
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    version='0.1.6',
    description='Library for sensible AI',
    install_requires=required_packages(open("requirements.txt").readlines()),
    dependency_links=["https://download.pytorch.org/whl/torch_stable.html"],
    setup_requires=["wheel"],
    extras_require={
        "torch": torch_requirements,
        "tensorflow": tf_requirements,
        "lightgbm": lightgbm_requirements,
        "geoanalytics": geoanalytics_requirements,
        "full": tf_requirements + torch_requirements + lightgbm_requirements + geoanalytics_requirements
    },
    author='jambit GmbH'
)