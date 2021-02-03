# sensAI

the Python library for sensible AI 

## About sensAI

sensAI provides a framework for AI and machine learning applications, integrating industry-standard libraries and providing additional abstractions that facilitate rapid implementation, experimentation as well as deployment. 

In particular, sensAI provides ...

* **machine learning** methods
  * **regression and classification** models
    * unified interface to models and algorithms of other machine learning libraries, particularly **scikit-learn**, **PyTorch** and **TensorFlow**
    * additional implementations of our own, e.g. for k-nearest neighbour models and naive Bayes models
  * mechanisms for **feature generation**, which serve to decouple externally provided input data from the data that is actually required as input to particular models
  * mechanisms for model-specific (input and output) **data transformation**, enabling, for example, convenient model-specific scaling/normalisation or encodings of features
  * (parallelised) **hyper-parameter optimisation** methods
  * **cloud-based tracking** of experimental results (with direct support for Microsoft Azure)
* **combinatorial optimisation**
  * **stochastic local search** methods, including (adaptive) simulated annealing and parallel tempering
* general utilities, including ...
  * extensive **caching mechanisms** (using SQLite, pickle and MySQL as backends)
  * multi-processing tools, e.g. a debugger for pickle errors

## Documentation

Source code documentation and tutorials can be found [here](https://sensai.readthedocs.io/)

### Integrating sensAI into a Project

sensAI may be integrated into your project in several ways: 

1. **Install it as a library** with `pip install sensai`.
   Choose this option if you do not intend to make changes to sensAI in the context of your project.
2. **Include sensAI's source code as a package within your project** (e.g. in `src/sensai`), which you synchronise with a sensAI branch.
   Choose this option if you intend to make changes to sensAI as you develop your project. When using this option, you (and others) may even make changes to sensAI in several branches of your project and even several projects using the same inclusion mechanism at the same time.
   See below for details on how synchronisation works.
3. **Clone sensAI and add its source directory to your `PYTHONPATH`**.
   Choose this option if you potentially intend to make changes to sensAI but no one else working on your project will do the same and you will be modifying sensAI's source in no more than one branch at a time.

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

## Contributors

sensAI is being developed by the artificial intelligence group at jambit GmbH.

The main contributors are Dominik Jain, Michael Panchenko, Kristof Schröder and Magnus Winter.

### How to contribute 

External contributions are welcome! Please issue a pull request.

#### Code Style

We deliberately do not comply with PEP 8 and do not intend to adapt in the future. 
If you decide to contribute, please strive for consistency.