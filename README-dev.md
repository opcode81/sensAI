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
     `git push`
     `git push --tags` (triggers PyPI release)

   If it it did not succeed and you need to fix stuff, 

   * Fix whatever you need to fix, adding commits
   * Create a new test build via
     `bumpversion build --commit`
   * Continue with step 3.

