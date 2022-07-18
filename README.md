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

Reference documentation and tutorials can be found [here](https://jambit.github.io/sensAI/docs/).

### Integrating sensAI into a Project

sensAI may be integrated into your project in several ways: 

1. **Install it as a library** with `pip install sensai`.
   Choose this option if you do not intend to make changes to sensAI in the context of your project.
2. **Include sensAI's source code as a package within your project** (e.g. in `src/sensai`), which you synchronise with a sensAI branch.
   Choose this option if you intend to make changes to sensAI as you develop your project. When using this option, you (and others) may even make changes to sensAI in several branches of your project and even several projects using the same inclusion mechanism at the same time.
   See developer documentation in README-dev.md for details on how synchronisation works.
3. **Clone sensAI and add its source directory to your `PYTHONPATH`**.
   Choose this option if you potentially intend to make changes to sensAI but no one else working on your project will do the same and you will be modifying sensAI's source in no more than one branch at a time.


## Contributors

sensAI is being developed by the artificial intelligence group at [jambit GmbH](http://www.jambit.com) and by members of [appliedAI](http://www.appliedai.de).

The main contributors are Dominik Jain, Michael Panchenko, Kristof Schröder and Magnus Winter.

### How to contribute 

External contributions are welcome! Please issue a pull request.

#### Code Style

We deliberately do not comply with PEP 8 and do not intend to adapt in the future. 
If you decide to contribute, please strive for consistency.
