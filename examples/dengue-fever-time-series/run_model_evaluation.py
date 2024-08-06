from typing import Optional

from matplotlib import pyplot as plt

from dengai.data import CITY_SAN_JUAN, CITY_IQUITOS, COL_WEEK_START_DATE, COL_TARGET
from dengai.models.model_factory import ModelFactory
from dengai.prediction_problem import DengaiPrediction
from sensai.evaluation.eval_util import RegressionMultiDataModelComparisonData
from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util import logging, mark_used
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag

mark_used(CITY_IQUITOS, CITY_SAN_JUAN, plt)


def eval_models(city_only: Optional[str] = None, use_cross_validation=False):
    pred = DengaiPrediction(city_only=city_only)

    experiment_name = f"dengAI-{pred.experiment_tag(use_cross_validation)}"
    run_id = datetime_tag()
    result_writer = ResultWriter(f"results/eval-{experiment_name}/{run_id}")
    logging.add_file_logger(result_writer.path("process.log"))

    week_window_size = pred.week_window_size
    mark_used(week_window_size)

    # define models to evaluate
    model_factories = []
    #model_factories.append(lambda: ModelFactory.create_baseline())
    #model_factories.append(lambda: ModelFactory.create_glm_benchmark())
    #model_factories.append(lambda: ModelFactory.create_rnn_mlp(week_window_size, batch_size=64))
    #model_factories.append(lambda: ModelFactory.create_rnn_mlp_autoreg(week_window_size))
    #model_factories.append(lambda: ModelFactory.create_rnn_mlp(week_window_size, feature_columns=ModelFactory.COLS_FEATURES_SEL))
    model_factories.append(lambda: ModelFactory.create_lstnet_encoder_decoder(week_window_size))
    #model_factories.append(lambda: ModelFactory.create_mlp(week_window_size))
    #model_factories.append(lambda: ModelFactory.create_xgb())
    #model_factories.append(lambda: ModelFactory.create_xgb_fsel())
    #model_factories.append(lambda: ModelFactory.create_xgb_window(week_window_size, min_child_weight=1))
    #model_factories.append(lambda: ModelFactory.create_xgb_window(week_window_size, min_child_weight=1, feature_columns=ModelFactory.COLS_FEATURES_SEL))
    #model_factories.append(lambda: ModelFactory.create_xgb_window_opt(week_window_size))
    #model_factories.append(lambda: ModelFactory.create_mean_past_year_week())

    # evaluate models
    ev = pred.create_multi_data_evaluator()
    results = ev.compare_models(model_factories, use_cross_validation=use_cross_validation, add_combined_eval_stats=True,
        create_metric_distribution_plots=False, create_combined_eval_stats_plots=False,
        result_writer=result_writer, write_per_dataset_results=True)
    results: RegressionMultiDataModelComparisonData

    # write metrics to mlflow
    tracking_exp = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_")
    for model_name in results.get_model_names():

        # track metrics
        eval_stats = results.get_eval_stats_collection(model_name).get_combined_eval_stats()
        with tracking_exp.begin_context(model_name, results.get_model_description(model_name)) as tracking_context:
            tracking_context.track_metrics(eval_stats.metrics_dict())

            # visualise prediction time series
            for dataset_name, result in results.iter_model_results(model_name):
                fig, ax = plt.subplots()
                full_io_data = pred.io_data_dict[dataset_name]
                plt.plot(full_io_data.inputs[COL_WEEK_START_DATE], full_io_data.outputs[COL_TARGET])
                for eval_data in result.iter_evaluation_data():
                    io_data = eval_data.io_data
                    predictions = eval_data.get_eval_stats().y_predicted
                    plt.plot(io_data.inputs[COL_WEEK_START_DATE], predictions)
                    fig_name = f"{model_name}_{dataset_name}_time-series"
                    result_writer.write_figure(fig_name, fig)
                    tracking_context.track_figure(fig_name, fig)


if __name__ == "__main__":
    logging.configure()

    use_cross_validation = False

    eval_models(city_only=CITY_IQUITOS, use_cross_validation=use_cross_validation)
    #eval_models(city_only=CITY_SAN_JUAN, use_cross_validation=use_cross_validation)
    #eval_models(use_cross_validation=use_cross_validation)
