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

### Using sensAI

There are two suggested ways of using sensai. 

1) Install it as a library with `pip install sensai`
2) Include the code in `src/sensai` as subpackage into your project.

The second way is recommended if you want to further develop/extend sensAI's functionality. In order to facilitate 
incorporating changes within sensAI into your code and vice versa, this repo contains a [syncing utility](repo_dir_sync.py).
Further documentation on that topic will be provided soon.

## Code Style

We deliberately do not comply with PEP 8. Requests to adhere to it will be ignored.

## Contributors

sensAI is being developed by the artificial intelligence group at jambit GmbH.

The main contributors are Dominik Jain, Michael Panchenko, Kristof Schröder and Magnus Winter.

### How to contribute 

External contributions are welcome! Please issue a pull request.

