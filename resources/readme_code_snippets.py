import math
from enum import Enum
from typing import Dict, Any
import os

from sklearn.preprocessing import StandardScaler

from sensai import InputOutputData
from sensai.data import DataSplitterFractional
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.evaluation import RegressionModelEvaluation, RegressionEvaluatorParams
from sensai.featuregen import MultiFeatureGenerator, FeatureGeneratorRegistry
from sensai.sklearn.sklearn_regression import SkLearnLinearRegressionVectorRegressionModel
from sensai.torch.torch_models.residualffn.residualffn_models import ResidualFeedForwardNetworkVectorRegressionModel
from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.xgboost import XGBRandomForestVectorRegressionModel

import random
import pandas as pd

# feature generators example

from sensai.featuregen import FeatureGeneratorMapColumn, FeatureGeneratorMapColumnDict, \
    FeatureGeneratorTakeColumns

class FeatureGeneratorTemperature(FeatureGeneratorTakeColumns):
    """
    Takes the input column "temperature" without modifications, adding meta-information
    on how to normalize/scale the feature (using StandardScaler)
    """
    def __init__(self):
        super().__init__("temperature",
            normalisation_rule_template=DFTNormalisation.RuleTemplate(
                transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler()))


class FeatureGeneratorWeekday(FeatureGeneratorMapColumn):
    """
    Creates the categorical feature "weekday" (integer from 0=Monday to 6=Sunday)
    from the "timestamp" column
    """
    def __init__(self):
        super().__init__(input_col_name="timestamp", feature_col_name="weekday",
            categorical_feature_names="weekday")

    def _create_value(self, timestamp: pd.Timestamp):
        return timestamp.weekday()


class FeatureGeneratorTimeOfDayCircular(FeatureGeneratorMapColumnDict):
    """
    From the "timestamp" column, creates two features "time_of_day_x" and
    "time_of_day_y", which correspond to the locations on the unit circle
    that the hour hand of a 24-hour clock would point to
    """
    def __init__(self):
        super().__init__(input_col_name="timestamp",
            normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True))

    def _create_features_dict(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        time_of_day_norm = (timestamp.hour + timestamp.minute / 60) / 24
        alpha = math.pi / 2 - time_of_day_norm * 2 * math.pi
        return dict(time_of_day_x=math.cos(alpha), time_of_day_y=math.sin(alpha))


class FeatureName(Enum):
    TEMPERATURE = "temperature"
    WEEKDAY = "weekday"
    TIME_OF_DAY_CIRC = "time_circ"


registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.TEMPERATURE, FeatureGeneratorTemperature)
registry.register_factory(FeatureName.WEEKDAY, FeatureGeneratorWeekday)
registry.register_factory(FeatureName.TIME_OF_DAY_CIRC, FeatureGeneratorTimeOfDayCircular)



if __name__ == '__main__':
    logging.configure()

    num_points = 200

    jan_2023 = 1672531200
    timestamps = [jan_2023, jan_2023+6*3600, jan_2023+12*3600, jan_2023+18*3600]
    for i in range(num_points):
        timestamps.append(jan_2023 + random.randint(0, 24*3600))

    temperatures = [20 + random.random() * 3 for _ in timestamps]

    df = pd.DataFrame({
        "timestamp": [pd.Timestamp(t, unit="s") for t in timestamps],
        "temperature": temperatures
    })

    targets = []
    for t in df.itertuples():
        ts: pd.Timestamp = t.timestamp
        result = 0
        if ts.hour >= 6 and ts.hour <= 16:
            result = t.temperature
        else:
            result = t.temperature - 2
        targets.append(result)

    df["target"] = targets

    fg = MultiFeatureGenerator(
        FeatureGeneratorWeekday(),
        FeatureGeneratorTimeOfDayCircular(),
        FeatureGeneratorTemperature())
    feature_df = fg.generate(df)

    # feature collector example

    feature_collector = registry.collect_features(
        FeatureName.TEMPERATURE,
        FeatureName.WEEKDAY)
    features_df = feature_collector.get_multi_feature_generator().generate(df)


    # DFT example

    feature_coll = registry.collect_features(*list(FeatureName))

    dft_normalization = feature_coll.create_dft_normalisation()
    dft_one_hot_encoder = feature_coll.create_dft_one_hot_encoder()


    # model example

    feature_coll = registry.collect_features(*list(FeatureName))

    model_xgb = XGBRandomForestVectorRegressionModel() \
        .with_name("XGBoost") \
        .with_feature_collector(feature_coll, shared=True) \
        .with_feature_transformers(
            feature_coll.create_dft_one_hot_encoder())
    model_linear = SkLearnLinearRegressionVectorRegressionModel() \
        .with_name("Linear") \
        .with_feature_collector(feature_coll, shared=True) \
        .with_feature_transformers(
            feature_coll.create_dft_one_hot_encoder())
    model_rffn = ResidualFeedForwardNetworkVectorRegressionModel(
            hidden_dims=[10]*5,
            cuda=False) \
        .with_name("RFFN") \
        .with_feature_collector(feature_coll, shared=True) \
        .with_feature_transformers(
        feature_coll.create_dft_one_hot_encoder(),
            feature_coll.create_dft_normalisation()) \
        .with_target_transformer(DFTSkLearnTransformer(StandardScaler()))

    # evaluation example

    io_data = InputOutputData.from_data_frame(df, "target")

    ev = RegressionModelEvaluation(io_data,
        RegressionEvaluatorParams(data_splitter=DataSplitterFractional(0.8)))

    ev.compare_models([model_xgb, model_linear, model_rffn])

    # tracking example

    experiment_name = "MyRegressionExperiment"
    run_id = datetime_tag()

    tracked_experiment = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_",
        add_log_to_all_contexts=True)

    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    ev.compare_models([model_xgb, model_linear, model_rffn],
        tracked_experiment=tracked_experiment,
        result_writer=result_writer)
