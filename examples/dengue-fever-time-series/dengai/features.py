from typing import Dict, Any, List, Optional

from .data import COL_GEN_WEEK_DP, WeekDataPoint, COLS_FEATURES, WeekDataPointInWindow
from sensai.data_transformation import DFTNormalisation
from sensai.featuregen import FeatureGeneratorMapColumn, FeatureGeneratorMapColumnDict

COL_FGEN_HISTORY_WINDOW = "history_window"
COL_FGEN_TARGET_WINDOW = "target_window"


class FeatureGeneratorHistoryWeekWindow(FeatureGeneratorMapColumn):
    def __init__(self, window_size: int, exclude_current: bool = False):
        super().__init__(COL_GEN_WEEK_DP, COL_FGEN_HISTORY_WINDOW,
            normalisation_rule_template=DFTNormalisation.RuleTemplate(unsupported=True))
        self.exclude_current = exclude_current
        self.window_size = window_size

    def _create_value(self, week_data_point: WeekDataPoint):
        w = week_data_point.window(self.window_size)
        if self.exclude_current:
            w = w[:-1]
        return w


class FeatureGeneratorTargetWeekWindow(FeatureGeneratorMapColumn):
    def __init__(self):
        super().__init__(COL_GEN_WEEK_DP, COL_FGEN_TARGET_WINDOW,
            normalisation_rule_template=DFTNormalisation.RuleTemplate(unsupported=True))

    def _create_value(self, week_data_point: WeekDataPoint):
        return week_data_point.window(1)


class FeatureGeneratorWindowColumnsFlat(FeatureGeneratorMapColumnDict):
    def __init__(self, window_size: int, columns: Optional[List[str]] = None):
        super().__init__(COL_GEN_WEEK_DP)
        self.window_size = window_size
        if columns is None:
            columns = COLS_FEATURES
        self.columns = columns

    def _create_features_dict(self, week_data_point: WeekDataPoint) -> Dict[str, Any]:
        window = week_data_point.window(self.window_size)
        result = {}
        for i, wdpiw in enumerate(window):
            wdpiw: WeekDataPointInWindow
            for col in self.columns:
                result[f"{col}_{i}"] = wdpiw.week_data_point.get_value(col)
        return result


class FeatureGeneratorTakeImputedColumns(FeatureGeneratorMapColumnDict):
    def __init__(self, columns: Optional[List[str]] = None):
        super().__init__(COL_GEN_WEEK_DP)
        if columns is None:
            columns = COLS_FEATURES
        self.columns = columns

    def _create_features_dict(self, week_data_point: WeekDataPoint) -> Dict[str, Any]:
        return {col: week_data_point.get_value(col, imputed=True) for col in self.columns}
