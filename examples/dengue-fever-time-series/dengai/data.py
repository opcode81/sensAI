import logging
from typing import Sequence, List, Hashable

import pandas as pd

from sensai import InputOutputData, RuleBasedDataFrameTransformer
from sensai.vectoriser import ItemIdentifierProvider, T

log = logging.getLogger(__name__)

COL_CITY = "city"
COL_YEAR = "year"
COL_WEEK_OF_YEAR = "weekofyear"
COL_WEEK_START_DATE = "week_start_date"

COL_FEATURE_STATION_MAX_TEMP = "station_max_temp_c"
COL_FEATURE_STATION_MIN_TEMP = "station_min_temp_c"
COL_FEATURE_STATION_AVG_TEMP = "station_avg_temp_c"
COL_FEATURE_STATION_PRECIP_MM = "station_precip_mm"
COL_FEATURE_STATION_DIURNAL_TEMP_RANGE = "station_diur_temp_rng_c"
COL_FEATURE_PERSIANN_PRECIPITATION_MM = "precipitation_amt_mm"
COL_FEATURE_REANALYSIS_SAT_PRECIP_MM = "reanalysis_sat_precip_amt_mm"
COL_FEATURE_REANALYSIS_DEW_POINT_K = "reanalysis_dew_point_temp_k"
COL_FEATURE_REANALYSIS_AIR_TEMP_K = "reanalysis_air_temp_k"
COL_FEATURE_REANALYSIS_REL_HUMIDITY_PERCENT = "reanalysis_relative_humidity_percent"
COL_FEATURE_REANALYSIS_SPEC_HUMIDITY = "reanalysis_specific_humidity_g_per_kg"
COL_FEATURE_REANALYSIS_PRECIP = "reanalysis_precip_amt_kg_per_m2"
COL_FEATURE_REANALYSIS_MAX_AIR_TEMP_K = "reanalysis_max_air_temp_k"
COL_FEATURE_REANALYSIS_MIN_AIR_TEMP_K = "reanalysis_min_air_temp_k"
COL_FEATURE_REANALYSIS_AVG_TEMP_K = "reanalysis_avg_temp_k"
COL_FEATURE_REANALYSIS_DIURNAL_TEMP_RANGE_K = "reanalysis_tdtr_k"
COL_FEATURE_VEG_INDEX_SE = "ndvi_se"
COL_FEATURE_VEG_INDEX_SW = "ndvi_sw"
COL_FEATURE_VEG_INDEX_NE = "ndvi_ne"
COL_FEATURE_VEG_INDEX_NW = "ndvi_nw"

# all numeric features
COLS_FEATURES = [COL_FEATURE_STATION_MAX_TEMP, COL_FEATURE_STATION_MIN_TEMP, COL_FEATURE_STATION_AVG_TEMP, COL_FEATURE_STATION_PRECIP_MM,
    COL_FEATURE_STATION_DIURNAL_TEMP_RANGE, COL_FEATURE_PERSIANN_PRECIPITATION_MM, COL_FEATURE_REANALYSIS_SAT_PRECIP_MM,
    COL_FEATURE_REANALYSIS_DEW_POINT_K, COL_FEATURE_REANALYSIS_AIR_TEMP_K, COL_FEATURE_REANALYSIS_REL_HUMIDITY_PERCENT,
    COL_FEATURE_REANALYSIS_SPEC_HUMIDITY, COL_FEATURE_REANALYSIS_PRECIP, COL_FEATURE_REANALYSIS_MAX_AIR_TEMP_K,
    COL_FEATURE_REANALYSIS_MIN_AIR_TEMP_K, COL_FEATURE_REANALYSIS_AVG_TEMP_K, COL_FEATURE_REANALYSIS_DIURNAL_TEMP_RANGE_K,
    COL_FEATURE_VEG_INDEX_SE, COL_FEATURE_VEG_INDEX_SW, COL_FEATURE_VEG_INDEX_NE, COL_FEATURE_VEG_INDEX_NW]
# features where the assumption of a normal distribution would be problematic
COLS_FEATURES_NONSTD = [COL_FEATURE_STATION_PRECIP_MM, COL_FEATURE_PERSIANN_PRECIPITATION_MM, COL_FEATURE_REANALYSIS_SAT_PRECIP_MM]
# features where the assumption of a normal distribution would be OK
COLS_FEATURES_STD = [c for c in COLS_FEATURES if c not in COLS_FEATURES_NONSTD]

CITY_IQUITOS = "iq"
CITY_SAN_JUAN = "sj"

COL_GEN_WEEK_DP = "week_data_point"

COL_TARGET = "total_cases"


class Dataset:
    def __init__(self):
        df = self._read_combined_df()
        self.df_sj = df[df[COL_CITY] == CITY_SAN_JUAN].reset_index()
        self.df_iq = df[df[COL_CITY] == CITY_IQUITOS].reset_index()
        self.df = df

    @staticmethod
    def _read_combined_df():
        # join training data
        train_features = pd.read_csv("data/dengue_features_train.csv")
        train_labels = pd.read_csv("data/dengue_labels_train.csv")
        train_df = pd.merge(train_features, train_labels, on=[COL_CITY, COL_YEAR, COL_WEEK_OF_YEAR])
        assert len(train_df) == len(train_labels)

        # concatenate with test data
        test_features = pd.read_csv("data/dengue_features_test.csv")
        df = pd.concat([train_df, test_features]).reset_index(drop=True)

        # obtain time stamps for week start date
        df[COL_WEEK_START_DATE] = pd.to_datetime(df[COL_WEEK_START_DATE])

        return df

    def get_city_data_frame(self, city: str) -> pd.DataFrame:
        if city == CITY_IQUITOS:
            return self.df_iq
        elif city == CITY_SAN_JUAN:
            return self.df_sj
        raise ValueError(city)

    def create_io_data(self, city: str, min_window_size: int, train=True):
        df = self.get_city_data_frame(city).copy(deep=True)

        if city == CITY_IQUITOS:
            df = df[df[COL_YEAR] >= 2002]  # remove seemingly broken data at the beginning of the Iquitos time series

        df.reset_index(drop=True, inplace=True)

        df_imputed = DFTImputation().apply(df)

        week_data_points = [WeekDataPoint(t_raw, t_imputed, df) for t_raw, t_imputed in zip(df.itertuples(), df_imputed.itertuples())]
        df[COL_GEN_WEEK_DP] = week_data_points

        if train:
            df = df.iloc[min_window_size:]

        if train:
            df = df[~df[COL_TARGET].isna()]
        else:
            df = df[df[COL_TARGET].isna()]

        return InputOutputData.from_data_frame(df, COL_TARGET)


class WeekDataPoint:
    def __init__(self, t_raw, t_imputed, df):
        self.t_raw = t_raw
        self.t_imputed = t_imputed
        self.df = df

    def window(self, size: int) -> Sequence["WeekDataPointInWindow"]:
        window = self.df[COL_GEN_WEEK_DP].iloc[self.t_raw.Index-(size-1):self.t_raw.Index+1].values
        assert window[-1] == self
        return [WeekDataPointInWindow(wdp, window, i) for i, wdp in enumerate(window)]

    def get_value(self, col: str, imputed: bool = False):
        if imputed:
            return getattr(self.t_imputed, col)
        else:
            return getattr(self.t_raw, col)

    def get_index(self) -> int:
        """
        :return: the index of the respective data point in the dataset (row index in DataFrame)
        """
        return self.t_raw.Index


class WeekDataPointInWindow:
    def __init__(self, week_data_point: WeekDataPoint, window: List[WeekDataPoint], idx: int):
        """
        :param week_data_point: the data point
        :param window: the full window
        :param idx: the 0-based index of the week data point within its window
        """
        self.week_data_point = week_data_point
        self.window = window
        self.idx = idx

    def is_current(self):
        return self.idx == len(self.window) - 1

    class IdentifierProvider(ItemIdentifierProvider["WeekDataPointInWindow"]):
        def get_identifier(self, item: "WeekDataPointInWindow") -> Hashable:
            return id(item.week_data_point)


class DFTImputation(RuleBasedDataFrameTransformer):
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Applying imputation")
        df = df.copy()
        for col in COLS_FEATURES:
            series = df[col]
            num_na = series.isna().sum()
            if num_na > 0:
                #log.debug(f"Imputing {num_na} values for column {col}")
                df[col] = series.fillna(method="ffill").fillna(method="bfill")
        return df
