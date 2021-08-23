# Build and Test Pipeline

The tests and docs build are executed via **tox** in several environments:
* `py`: the "regular" test environment, where we test against the pinned dependencies which we also use for development (by explicitly including `requirements.txt` with the pinned versions; this is also the environment in which we test the execution of notebooks
* `py_latest_dependencies`: the environment where we use the latest versions of all dependencies (except where we have identified an incompatibility; see `setup.py` definitions `DEPS_VERSION_LOWER_BOUND` and `DEPS_VERSION_UPPER_BOUND_EXCLUSIVE`); by not including `requirements.txt`, we depend on the latest admissible versions according to `setup.py`
* `docs`: the environment in which docs are built via sphinx (by executing `build_scripts/update_docs.py`)

## Docs Build

Docs are automatically created, all .rst files are auto-generated; only `index.rst` is manually defined.

Make sure that any optional sensAI dependencies (which are not included in the `docs` tox environment) are added to `docs/conf.py` under `autodoc_mock_imports`.

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

