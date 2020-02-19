from setuptools import find_packages, setup

test_requirements = ['pytest']
docs_requirements = ['Sphinx==2.4.2', 'sphinxcontrib-websupport==1.2.0', 'sphinx_rtd_theme']

setup(
    name='sensai',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version='0.0.1',
    description='Library for sensible AI',
    install_requires=open("requirements.txt").readlines(),
    setup_requires=["wheel"],
    tests_require=test_requirements,
    extras_require={
        "test": test_requirements,
        "docs": docs_requirements
    },
    author='jambit GmbH'
)
