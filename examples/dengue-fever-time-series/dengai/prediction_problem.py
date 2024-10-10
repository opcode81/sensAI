from typing import List

from .data import Dataset, CITY_SAN_JUAN, CITY_IQUITOS
from sensai.evaluation import MultiDataModelEvaluation, VectorModelCrossValidatorParams, \
    RegressionEvaluatorParams
from sensai.evaluation.crossval import CrossValidationSplitterNested
from sensai.evaluation.eval_stats import RegressionMetricMAE
from sensai.evaluation.metric_computation import MetricComputationMultiData
from sensai.util.string import ToStringMixin, TagBuilder


class DengaiPrediction(ToStringMixin):
    def __init__(self, week_window_size: int = 12, city_only=None, test_fraction=0.2, cv_folds=3, cv_test_fraction=0.2):
        self.test_fraction = test_fraction
        self.cv_test_fraction = cv_test_fraction
        self.cv_folds = cv_folds
        self.city_only = city_only
        self.week_window_size = week_window_size
        self.dataset = Dataset()
        self.evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=test_fraction,
            fractional_split_shuffle=False)
        self.cross_validator_params = VectorModelCrossValidatorParams(folds=cv_folds,
            splitter=CrossValidationSplitterNested(cv_test_fraction))
        self.io_data_dict = {city: self.dataset.create_io_data(city, min_window_size=self.week_window_size)
            for city in [CITY_SAN_JUAN, CITY_IQUITOS]
            if city_only is None or city_only == city}

    def experiment_tag(self, use_cross_validation: bool):
        return TagBuilder().with_alternative(self.city_only is None, "combined", self.city_only) \
            .with_conditional(use_cross_validation, f"cv{self.cv_folds}-{self.cv_test_fraction}") \
            .with_conditional(not use_cross_validation and self.test_fraction != 0.2, f"split{self.test_fraction}") \
            .build()

    def _tostring_excludes(self) -> List[str]:
        return ["io_data_dict"]

    def create_multi_data_evaluator(self):
        return MultiDataModelEvaluation(self.io_data_dict, key_name="city", evaluator_params=self.evaluator_params,
            cross_validator_params=self.cross_validator_params)

    def create_metric_computation(self, use_cross_validation: bool):
        ev = self.create_multi_data_evaluator()
        return MetricComputationMultiData(ev, use_cross_validation=use_cross_validation, metric=RegressionMetricMAE(),
            use_combined_eval_stats=True)
