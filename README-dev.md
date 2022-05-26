# Development Environment

This section explains the steps required to set up an environment in order to develop sensAI further.

## Clone Large Files

Clone the full repo, including large files using [git LFS](https://git-lfs.github.com):

    git lfs pull

This adds, in particular, data that is used in notebooks.

## Create the Python Virtual Environment

Use conda to set up the Python environment:

    conda env create -f environment.py

Solving the environment may take several minutes (but should ultimately work).

NOTE: versions are mostly unpinned in the environment specification, because this facilitates conda dependency resolution. Also, sensAI is intended to be compatible with all (newer) versions of the dependencies. If it isn't, we need to specify  an upper version bound in `setup.py` (where it matters the most) as well as in `environment.yml`. Compatibility with old (pinned) versions and the latest versions is tested in the tox build (see below).

# Build and Test Pipeline

The tests and docs build are executed via **tox** in several environments:
* `py`: the "regular" test environment, where we test against the pinned dependencies which we also use for development (by explicitly including `requirements.txt` with the pinned versions; this is also the environment in which we test the execution of notebooks
* `py_latest_dependencies`: the environment where we use the latest versions of all dependencies (except where we have identified an incompatibility; see `setup.py` definitions `DEPS_VERSION_LOWER_BOUND` and `DEPS_VERSION_UPPER_BOUND_EXCLUSIVE`); by not including `requirements.txt`, we depend on the latest admissible versions according to `setup.py`
* `docs`: the environment in which docs are built via sphinx 

## Docs Build

Docs are automatically created during the GitHub build via tox.

All .rst files are auto-generated (by `build_scripts/update_docs.py`), with the exception of the root index file  `index.rst`.

### Dependency Handling

**Attention**: Make sure that any optional sensAI dependencies (which are not included in the `docs` tox environment) are added to `docs/conf.py` under `autodoc_mock_imports`.

### Notebooks

Notebooks in the `notebooks/` folder are executed and saved with outputs to the `docs/` folder by a test in `notebooks/test_notebooks.py`.

Notebooks are included in the documentation in `docs/index.rst`.

### Manually Running the Docs Build

The docs build is designed to be run by tox. As soon as a single iPython notebook is included in index.rst, `sensai` must be available as an installed dependency in the environment in which the docs build is run, as the notebook executions take place in a Jupyter environment where we can't simply extend sys.path. 

Under Linux, running tox should work fine. Under Windows, we are yet to succeed.

To run the docs build without tox (under Windows), first create an environment that has the additional requirements installed.

    conda env create -n sensai-docs -f environment.yml
    conda activate sensai-docs
    pip install sphinx sphinx_rtd_theme nbsphinx

Then, to build the docs, we need to perform the following steps with the new environment `sensai-docs` activated. The first command installs the current source version of  `sensai` itself.

    pip install .
    rm -rf docs-build; mkdir docs-build
    sphinx-build -b html docs docs-build

# Creating a New Release

1. Switch to the `master` branch and merge any content the new release is to contain

2. Bump the version that the new release shall change by using one of the following commands:

   * `bumpversion patch  --commit`
   * `bumpversion minor --commit`
   * `bumpversion major  --commit`

   This will create a new "-dev" version which can be pushed without a release ending up on PyPI.

3. Push this version to github
   `git push`
   and then check whether tests pass and the build succeeds.

4. If the build succeeded and you want to release this version, 

   * Create the release version:
     `bumpversion release --commit --tag`
   * Push the new release:
     * `git push`
     * `git push --tags` (triggers PyPI release)

   If it it did not succeed and you need to fix stuff, 

   * Fix whatever you need to fix, adding commits
   * Create a new test build via
     `bumpversion build --commit`
   * Continue with step 3.

# Source-Level Directory Sync

#### Details on the Synchonisation of a Source Directory within Your Project with the sensAI Repository

We support the synchronisation of a branch in the sensAI repository with a directory within the git repository of your project which is to contain the sensAI source code (i.e. alternative #2 from above) via a convenient scripting solution.

We consider two local repositories: the sensAI repository in directory `sensAI/` and your project in, for instance, directory `sensAI/../myprj/`. Let us assume that we want to synchronise branch `myprj-branch` in the sensAI repository with directory `myprj/src/sensai`.

##### Synchronisation Script

To perform the synchronisation, please create a script as follows, which you should save to `sensAI/sync.py`:

```python
import os
from repo_dir_sync import LibRepo, OtherRepo

r = LibRepo()
r.add(OtherRepo("myprj", "myprj-branch", os.path.join("..", "myprj", "src", "sensai")))
r.runMain()
```

You can add multiple other repositories if you so desire in the future.

From directory `sensAI/` you can use the script in order to 

* ***Push***: Update your project (i.e. `myprj/src/sensai`) with changes that were made in other projects by running `python sync.py myprj push`
* ***Pull***: Update `myprj-branch` in the sensAI repository with changes made in your project by running `python sync.py myprj pull`

##### Initialisation

To initialise the synchronisation, proceed as follows:

1. Create the branch `myprj-branch` in the sensAI repository, i.e. in `sensAI/` run this command:
   `git branch myprj-branch master`
2. Create the directory `myprj/src/sensai`.
3. Make sure you have a `.gitignore` file in `myprj/` with at least the following entries:

       *.pyc
       __pycache__
       *.bak
       *.orig

   Otherwise you may end up with unwanted tracked files after a synchronisation.
4. Perform the initial *push*, i.e. in `sensAI/` run this command:
   `python sync.py myprj push`

##### Things to Keep in Mind

* Both *push* and *pull* operations are always performed based on the branch that is currently checked out in `myprj/`. The best practice is to only use one branch for synchronisation, e.g. master.
* *Push* and *pull* operations will make git commits in both repositories. Should an operation ever go wrong/not do what you intended, use `git reset --hard` to go back to the commits before the operation in both repositories.