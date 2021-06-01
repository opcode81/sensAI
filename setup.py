from setuptools import find_packages, setup

tf_requirements = ['tensorflow==1.15.0']
torch_requirements = ['torch==1.4.0', 'torchtext==0.5.0']
lightgbm_requirements = ['lightgbm==2.3.0']
geoanalytics_requirements = ['networkx==2.4', 'Shapely~=1.7.0', 'geopandas==0.7.0']

setup(
    name='sensai',
    package_dir={"": "src"},
    license="MIT",
    url="https://github.com/jambit/sensAI",
    packages=find_packages(where="src"),
    include_package_data=True,
    version='0.1.1',
    description='Library for sensible AI',
    install_requires=[
        line
        for line in open("requirements.txt").readlines()
        if not line.startswith("--")
    ],
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
