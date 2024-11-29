# Changelog


## Unreleased

### Improvements/Changes

* Add extra `xgboost` on PyPI. sensAI supports a wide range of XGBoost versions (dating back to 2020), 
  but with the extra, we opted to use 1.7 as a lower bound, as compatibility with this version 
  is well-tested. 
* `util`:
  * `util.version`: Add methods `Version.is_at_most` and `Version.is_equal` 
  * `util.logging`: 
    * `add_memory_logger` now returns the logger instance, which can be queried to retrieve the log (see breacking
      change below)
    * Add class `MemoryLoggerContext`, which be used in conjunction with Python's `with` statement to record logs
* `evaluation`:
  * `EvaluationResultCollector`: Add method `is_plot_creation_enabled`
* `data`:
  * `InputOutputData`: Add method `to_df`
  * Add module `data.dataset` containing sample datasets (mainly for demonstration purposes)
* `tracking`:
  * `mlflow_tracking`: Option `add_log_to_all_contexts` now stores only the logs of each model's training process (instead of the entire 
    process beginning with the instantiation of the experiment) 
### Breaking Changes:

* `util.logging`: Change `add_memory_logger` to no longer define a global logger, but return the handler (an instance of  
  `MemoryStramHandler`) instead. Consequently removed method `get_memory_log` as it is no longer needed (use the handler's method 
  `get_log` instead). 

### Fixes:

* `evaluation`:
  * `ModelEvaluation` (and subclasses): Fix plots being shown if no `ResultWriter` is used
  even though `show_plots=False`
* `vector_model`:
  * `VectorModel`: Fix data frame transformers not appearing in string representations
* `data_transformation`:
  * `DFTOneHotEncoder`: Fitting failed in the presence of missing values


## v1.2.1 (2024-08-10)

### Improvements/Changes

* `util`
  * Minimise required dependencies for all modules in this package in preparation of the release of *sensAI-utils* 
* `util.logging`:
  * Fix type annotations of `run_main` and `run_cli` 


## v1.2.0 (2024-06-10)

### Improvements/Changes

* `util.cache`:
  * Add new base class `KeyValueCache` alongside `PersistentKeyValueCache`
  * Add `InMemoryKeyValueCache`
  * `PickleCached` 
    * Rename to `pickle_cached`, keeping old name as alias  
    * Change implementation to use nested functions instead of a class to improve IDE support 
    * Auto-create the storage directory if it does not yet exist
  * Support `cloudpickle` as a backend 
* `columngen`:
  * `ColumnGenerator`: add method `to_feature_generator`
* `evaluation`:
  * `MultiDataEvaluation`: Add option to supply test data (without using splitting)
  * `VectorRegressionModelEvaluator`: Handle output column mismatch between model output and ground truth 
    for the case where there is only a single column, avoiding the exception and issuing a
    warning instead
* `dft`:
  * `DFTNormalisation.RuleTemplate`: Add attributes `fit` and `array_valued`
* `util.deprecation`: Apply `functools.wrap` to retain meta-data of wrapped function
* `util.logging`: 
  * Support multiple configuration callbacks in `set_configure_callback` 
  * Add line number to default format (`LOG_DEFAULT_FORMAT`)
  * Add function `is_enabled` to check whether a log handler is registered
  * Add context manager `LoggingDisabledContext` to temporarily disable logging
  * Add `FallbackHandler` to support logging to a fallback destination (if no other handlers are defined) 
* `util.io`:
  * `ResultWriter`:
    * Allow to disable an instance such that no results are written (constructor parameter `enabled`)
    * Add default configuration for closing figures after writing them (constructor parameter `close_figures`)
    * `write_image`: Improve layout in written images by setting `bbox_inches='tight'`


## v1.1.0 (2024-02-19)

### Improvements/Changes

* `vectoriser`:
  * `SequenceVectoriser`: 
    * Allow to inject a sequence item identifier provider
      (instance of new class `ItemIdentifierProvider`) in order to determine the set of
      relevant unique items when using fitting mode UNIQUE
    * Allow sharing of vectorisers between instances such
      that a previously fitted vectoriser can be reused in its fitted state,
      which can be particularly useful for encoder-decoder settings where
      the decoding stage uses some of the same features (vectorisers) as the
      encoding stage.
  * Make Vectorisers aware of their 'fitted' status.
* `torch`:
  * `TorchVectorRegressionModel`: Add support for auto-regressive predictions
    by adding class `TorchAutoregressiveResultHandler` and method 
    `with_autogressive_result_handler`
  * `LSTNetwork`:
    * Add new mode 'encoder', where the output of the complex path
      prior to the dense layer is returned
    * Changed constructor interface to comply with PEP-8
  * Add package `seq` for encoder-decoder-style sequence models, adding the
    highly flexible vector model implementation 
    `EncoderDecoderVectorRegressionModel` and a multitude of low-level encoder 
    and decoder modules
* `data`:
  * Add `DataFrameSplitterColumnEquivalenceClass`, which splits a data frame
    based on equivalence classes of a given column
* `evaluation`:
  * `ModelEvaluation` (and derived classes): Support direct specification of the test data  
    (previously only indirect specification via a splitter was supported)

#### Breaking Changes

* `GridSearch`: Change return value to a result object for convenient retrieval

### Fixes

* `TagBuilder`: Fix return value of `with_component`  
* `ModelEvaluation`: `create_plots` did not track plots with given tracking context
   if `show_plots`=False and `result_writer`=None. 
* `ParametersMetricsCollection`: `csv_path` could not be None 
* `LSTNetworkVectorClassificationModel` is now functional in v1,
  improving the representation (no more dictionaries).
  This breaks compatibility with sensAI v0.x representations of this class.


## v1.0.0 (2023-12-06) 

### Improvements/Changes

* `tracking`:
   * Improve (under-the-hood) tracking interfaces, introducing the concept of a tracking
     context (class `TrackingContext`, which is typically model-specific) in addition to the more
     high-level 'experiment' concept
   * Full support for cross-validation
   * Adapt & improve MLflow tracking implementation
* `util.datastruct`:
    * `SortedKeysAndValues`, `SortedKeyValuePairs`: Add `__len__`
* `featuregen`:
    * `FeatureCollector`: Add factory methods for the generation of
      DFTNormalisation and DFTOneHotEncoder instances (for convenience)
    * `FeatureGeneratorRegistry`:
        * Improve type annotation of singleton dictionary
        * Add convenience method `collect_features`, which creates a
          FeatureCollector
* `util.io`:
    * `write_data_frame_csv_file`: Add options `index` and `header`
* `util.pickle`:
    * `dump_pickle`, `load_pickle`: `PickleLoadSaveMixin`: Support passing `Path` objects
* `vector_model`:
    *  Pre-processors are now included in models string representations by default 
* `torch`:
    * `TorchVector*Model`: Improve type hints for with* methods 
* `evaluation`:
    * `MultiDataModelEvaluation` (previously `MultiDataEvaluationUtil`):
       * Add model description/string representation to result object
    * Add class `CrossValidationSplitterNested` (for nested cross-validation)
    * `ModelComparisonData.Result`: Add method `iter_evaluation_data`
* `feature_selection`:
    * Add `RecursiveFeatureElimination` (to complement existing CV-based implementation)
* `util.string`:
    * Add class `TagBuilder` (for generation of dataset/experiment tags/identifiers)
* `util.logging`:
    * Add in-memory logging (`add_memory_logger`, `get_memory_log`)
    * Reuse configured log format (if any) for both file & in-memory loggers
    * Add functions `run_main` and `run_cli` for convenient setup
    * Add `set_configure_callback` for third-party usage of `configure`, allowing
      users to add additional configuration via a callback
    * Add `remove_log_handler`
    * Add `FileLoggerContext` for file-based logging within a `with`-block
* Refactoring:
    * Module `featuregen` is now a package with modules
       * `feature_generator` (all feature generators)
       * `feature_generator_registry` (registry and feature collector) 
* Testing:
    * Add test for typical usage of `FeatureCollector` in conjunction with
      `FeatureGeneratorRegistry`  

### Breaking Changes

* Changed *all* camel case interfaces (methods and parameters) as well as
  local variables to use snake case in order to align more closely with PEP 8.

  This breaks source-level compatibility with earlier v0 releases.
  However, persisted objects from earlier versions should still be loadable,
  as attribute names in classes that may have been persisted remain in
  camel case. Strictly speaking, PEP 8 makes no statement about the
  format of attribute names, so there is not really a violation anyway.
*  Removed some deprecated interfaces (particularly support for the
   kwargs/dict interface in parallel to parameter objects in evaluators)
* `TorchVector*Model`: Changed construction of contained `TorchModel` to a 
  no-args factory (i.e. support for `modelArgs` and `modelKwArgs` dropped). 
  The new mechanism is both simpler and does not encourage usage patterns 
  where correct construction cannot be statically checked (in contrast to the 
  old mechanism). The new mechanisms encourages the implementation of
  dedicated factory methods (but could be abused with `functools.partial`,
  of course).
* `FeatureGeneratorRegistry`:
  Removed support for discouraged mechanism of setting/getting feature
  generator factories via `__setattr__`/`__getattr__`
* `NNOptimiserParams`: Do not use kwargs for parameters to be passed on 
  to the underlying optimiser, use dict `optimiser_args` instead
* `MultiDataModelEvaluation` (previously `MultiDataEvaluationUtil`):
  * Moved evaluator and cross-validator params to constructor
  * Removed deprecated method `compare_models_cross_validation`
* `RegressionEvalStats`: Rename methods using inappropriate prefix `get` (now `compute`)
* Renamed high-level evaluation classes:
   * `RegressionEvalUtil` renamed to `RegressionModelEvaluation`
   * `ClassificationEvalUtil` renamed to `ClassificationModelEvaluation`
   * `MultiDataEvaluationUtil` renamed to `MultiDataModelEvaluation`
   * `Vector*ModelEvaluatorParams` -> `*EvaluatorParams`
* Changed default parameters of `SkLearnDecisionTreeVectorClassificationModel`
  and `SkLearnRandomForestVectorClassificationModel` to align with sklearn
  defaults

### Fixes

* `ToStringMixin`:
  Prevent infinite recursion for case where ToStringMixin references a bound
  method of itself
* `TorchVectorModels`: Dropped support for model kwargs in constructor
* `MultiDataModelEvaluation` (previously `MultiDataEvaluationUtil`): 
  * dataset key column was not removed prior to mean computation (would fail if 
    value is non-numeric)
  * Combined eval stats were not logged
* `EvalStatsClassification`: Do not attempt to create precision/recall plots if
  class probabilities are unavailable


## v0.2.0 (2023-07-20)

Final pre-release (primarily for internal use at jambit GmbH
and appliedAI Initiative GmbH)


## Earlier Pre-Releases (2020-2022)

* v0.1.9 (2022-07-20)
* v0.1.8 (2022-07-01)
* v0.1.7 (2022-02-22)
* v0.1.6 (2021-07-16)
* v0.1.5 (2021-06-22)
* v0.1.4 (2021-06-21)
* v0.1.1 (2021-06-01)
* v0.1.0 (2021-05-25)
* v0.0.8 (2021-02-18)
* v0.0.4 (2020-10-16)
* v0.0.1 (2020-02-20)

