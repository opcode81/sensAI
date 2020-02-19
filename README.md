# sensAI

the library for sensible AI by jambit GmbH.

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

Additional documentation will be provided in the near future. Stay tuned.

For now, we refer to the docstrings within the source code.

## Code Style

We deliberately do not comply with PEP 8. Requests to adhere to it will be ignored.

## Contributors

sensAI is being developed by the artificial intelligence group at jambit GmbH.

The main contributors are Dominik Jain, Michael Panchenko, Kristof Schröder and Magnus Winter.

External contributions are welcome! Please issue a pull request.

