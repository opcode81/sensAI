from typing import Optional

from ..data import *
from ..features import FeatureGeneratorWindowColumnsFlat, FeatureGeneratorHistoryWeekWindow, FeatureGeneratorTargetWeekWindow, \
    COL_FGEN_HISTORY_WINDOW, COL_FGEN_TARGET_WINDOW
from .glm import GeneralisedLinearModel
from .mean_week_model import MeanPastYearWeekModel
from .rnn_mlp import RnnMlp
from ..vectorisers import create_history_sequence_vectoriser, AutoregressiveResultHandler, create_target_sequence_vectoriser
from sensai.data_transformation import SkLearnTransformerFactoryFactory, DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns, MultiFeatureGenerator
from sensai.sklearn.sklearn_regression import SkLearnDummyVectorRegressionModel
from sensai.torch import Optimiser, NNOptimiserParams, NNLossEvaluatorRegression
from sensai.torch.torch_models.seq.seq_models import EncoderDecoderVectorRegressionModel
from sensai.torch.torch_models.seq.seq_modules import LSTNetworkEncoderFactory, SingleTargetDecoderFactory, LinearPredictorFactory
from sensai.xgboost import XGBGradientBoostedVectorRegressionModel


class ModelFactory:
    COLS_FEATURES_SEL = [COL_FEATURE_STATION_MAX_TEMP, COL_FEATURE_STATION_MIN_TEMP, COL_FEATURE_STATION_AVG_TEMP,
        COL_FEATURE_REANALYSIS_PRECIP, COL_FEATURE_REANALYSIS_DEW_POINT_K, COL_FEATURE_REANALYSIS_SPEC_HUMIDITY]

    @classmethod
    def create_baseline(cls):
        return SkLearnDummyVectorRegressionModel().with_name("Baseline")

    @classmethod
    def create_rnn_mlp(cls, week_window_size: int, feature_columns: Optional[List[str]] = None, auto_regressive=False, batch_size=8,
            optimiser_lr=1e-5, loss_fn: NNLossEvaluatorRegression.LossFunction = NNLossEvaluatorRegression.LossFunction.SMOOTHL1LOSS,
            early_stopping_epochs=100):
        if feature_columns is None:
            feature_columns = COLS_FEATURES
        return RnnMlp(
                NNOptimiserParams(optimiser=Optimiser.ADAMW,
                    early_stopping_epochs=early_stopping_epochs,
                    optimiser_lr=optimiser_lr,
                    batch_size=batch_size,
                    train_fraction=0.9,
                    use_shrinkage=True,
                    loss_evaluator=NNLossEvaluatorRegression(loss_fn)),
                rnn_hidden_dim=32,
                mlp_hidden_dims=(32, 16),
                mlp_dropout=0.2,
                week_window_size=week_window_size,
                feature_columns=feature_columns,
                auto_regressive=auto_regressive) \
            .with_name("RnnMlp")

    @classmethod
    def create_rnn_mlp_autoreg(cls, week_window_size: int, feature_columns: Optional[List[str]] = None):
        return cls.create_rnn_mlp(week_window_size, feature_columns=feature_columns, batch_size=64,
                auto_regressive=True) \
            .with_name("RnnMlpAutoReg")

    @classmethod
    def create_lstnet_encoder_decoder(cls, week_window_size: int):
        autoregressive_result_handler = AutoregressiveResultHandler()
        target_transformer_factory = SkLearnTransformerFactoryFactory.MaxAbsScaler()
        history_sequence_vectoriser = create_history_sequence_vectoriser(COLS_FEATURES, True, target_transformer_factory,
            autoregressive_result_handler)
        target_sequence_vectoriser = create_target_sequence_vectoriser(history_sequence_vectoriser)
        encoder_factory = LSTNetworkEncoderFactory(week_window_size-1, 100, 4, 32, 0, 0)
        decoder_factory = SingleTargetDecoderFactory(LinearPredictorFactory())
        # decoder_factory = SingleTargetDecoderFactory(MLPPredictorFactory(output_activation_fn=ActivationFunction.RELU))
        nn_optimiser_params = NNOptimiserParams(batch_size=64, shuffle=True)
        latent_dim = encoder_factory.get_latent_dim()
        return EncoderDecoderVectorRegressionModel(False,
                    COL_FGEN_HISTORY_WINDOW, history_sequence_vectoriser, False,
                    COL_FGEN_TARGET_WINDOW, target_sequence_vectoriser,
                    latent_dim,
                    encoder_factory,
                    decoder_factory,
                    nn_optimiser_params=nn_optimiser_params) \
            .with_feature_generator(MultiFeatureGenerator(
                FeatureGeneratorHistoryWeekWindow(week_window_size, exclude_current=True),
                FeatureGeneratorTargetWeekWindow())) \
            .with_autoregressive_result_handler(autoregressive_result_handler) \
            .with_target_transformer(DFTSkLearnTransformer(target_transformer_factory())) \
            .with_name("LSTNetEncoderDecoder")

    @classmethod
    def create_mlp(cls, week_window_size: int, feature_columns: Optional[List[str]] = None):
        if feature_columns is None:
            feature_columns = COLS_FEATURES
        return RnnMlp(
                NNOptimiserParams(optimiser=Optimiser.ADAMW, early_stopping_epochs=30, optimiser_lr=1e-5, batch_size=8, train_fraction=0.9,
                    use_shrinkage=True,
                    loss_evaluator=NNLossEvaluatorRegression(NNLossEvaluatorRegression.LossFunction.SMOOTHL1LOSS)),
                use_rnn=False,
                rnn_hidden_dim=32,
                mlp_hidden_dims=(32, 16),
                mlp_dropout=0.2,
                week_window_size=week_window_size,
                feature_columns=feature_columns) \
            .with_name("Mlp")

    @classmethod
    def create_xgb(cls, feature_columns: Optional[List[str]] = None, min_child_weight=5, **kwargs):
        if feature_columns is None:
            feature_columns = COLS_FEATURES
        return XGBGradientBoostedVectorRegressionModel(min_child_weight=min_child_weight, objective="reg:absoluteerror", **kwargs) \
            .with_feature_generator(FeatureGeneratorTakeColumns(feature_columns)) \
            .with_name("XGBoost")

    @classmethod
    def create_xgb_fsel(cls):
        feature_columns = ['station_min_temp_c', 'station_avg_temp_c', 'station_precip_mm', 'precipitation_amt_mm', 'reanalysis_sat_precip_amt_mm', 'reanalysis_dew_point_temp_k', 'reanalysis_air_temp_k', 'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']
        params = {'colsample_bytree': 0.2867610159931737, 'gamma': 8.62166097248104, 'max_depth': 4, 'min_child_weight': 12.0,
            'reg_lambda': 0.7172266188561198}
        return XGBGradientBoostedVectorRegressionModel(objective="reg:absoluteerror", **params) \
            .with_feature_generator(FeatureGeneratorTakeColumns(feature_columns)) \
            .with_name("XGBoost-fsel")

    @classmethod
    def create_xgb_window(cls, window_size: int, min_child_weight=5, feature_columns: Optional[List[str]] = None, **kwargs):
        if feature_columns is None:
            feature_columns = COLS_FEATURES
        return XGBGradientBoostedVectorRegressionModel(min_child_weight=min_child_weight, objective="reg:absoluteerror", **kwargs) \
            .with_feature_generator(FeatureGeneratorWindowColumnsFlat(window_size, feature_columns)) \
            .with_name("XGBoostWin")

    @classmethod
    def create_xgb_window_opt(cls, window_size=12):
        params = {'colsample_bytree': 0.2867610159931737, 'gamma': 8.62166097248104, 'max_depth': 4, 'min_child_weight': 12.0,
            'reg_lambda': 0.7172266188561198}
        return cls.create_xgb_window(window_size=window_size, **params)

    @classmethod
    def create_mean_past_year_week(cls):
        return MeanPastYearWeekModel().with_name("MeanPastYearWeek")

    @classmethod
    def create_glm_benchmark(cls):
        return GeneralisedLinearModel()
