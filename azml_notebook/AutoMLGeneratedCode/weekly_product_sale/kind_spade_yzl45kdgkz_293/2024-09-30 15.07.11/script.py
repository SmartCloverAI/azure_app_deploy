# This file has been autogenerated by version 1.57.0 of the Azure Automated Machine Learning SDK.


import numpy
import numpy as np
import pandas as pd
import pickle
import argparse


# For information on AzureML packages: https://docs.microsoft.com/en-us/python/api/?view=azure-ml-py
from azureml.training.tabular._diagnostics import logging_utilities


def setup_instrumentation(automl_run_id):
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Add logging to STDOUT
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        # Add telemetry logging with formatter to strip identifying info
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)

        # Attach run IDs to logging info for correlation if running inside AzureML
        try:
            run = Run.get_context()
            return logging.LoggerAdapter(logger, extra={
                "properties": {
                    "codegen_run_id": run.id,
                    "automl_run_id": automl_run_id
                }
            })
        except Exception:
            pass
    except Exception:
        pass

    return logger


automl_run_id = 'kind_spade_yzl45kdgkz_293'
logger = setup_instrumentation(automl_run_id)


def split_dataset(X, y, weights, split_ratio, should_stratify):
    '''
    Splits the dataset into a training and testing set.

    Splits the dataset using the given split ratio. The default ratio given is 0.25 but can be
    changed in the main function. If should_stratify is true the data will be split in a stratified
    way, meaning that each new set will have the same distribution of the target value as the
    original dataset. should_stratify is true for a classification run, false otherwise.
    '''
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_uri):
    
    from azureml.core.run import Run
    from azureml.data.abstract_dataset import AbstractDataset
    
    logger.info("Running get_training_dataset")
    ws = Run.get_context().experiment.workspace
    dataset = AbstractDataset._load(dataset_uri, ws)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    '''
    Prepares data for training.
    
    Cleans the data, splits out the feature and sample weight columns and prepares the data for use in training.
    This function can vary depending on the type of dataset and the experiment task type: classification,
    regression, or time-series forecasting.
    '''
    
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'CANTITATE'
    
    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=True, target_column=label_column_name)
    
    return X, y, sample_weights


def generate_data_transformation_config():
    from azureml.training.tabular.featurization._featurization_config import FeaturizationConfig
    from azureml.training.tabular.featurization.timeseries.category_binarizer import CategoryBinarizer
    from azureml.training.tabular.featurization.timeseries.drop_columns import DropColumns
    from azureml.training.tabular.featurization.timeseries.grain_index_featurizer import GrainIndexFeaturizer
    from azureml.training.tabular.featurization.timeseries.missingdummies_transformer import MissingDummiesTransformer
    from azureml.training.tabular.featurization.timeseries.numericalize_transformer import NumericalizeTransformer
    from azureml.training.tabular.featurization.timeseries.restore_dtypes_transformer import RestoreDtypesTransformer
    from azureml.training.tabular.featurization.timeseries.short_grain_dropper import ShortGrainDropper
    from azureml.training.tabular.featurization.timeseries.time_index_featurizer import TimeIndexFeaturizer
    from azureml.training.tabular.featurization.timeseries.time_series_imputer import TimeSeriesImputer
    from azureml.training.tabular.featurization.timeseries.timeseries_transformer import TimeSeriesPipelineType
    from azureml.training.tabular.featurization.timeseries.timeseries_transformer import TimeSeriesTransformer
    from azureml.training.tabular.featurization.timeseries.unique_target_grain_dropper import UniqueTargetGrainDropper
    from collections import OrderedDict
    from numpy import dtype
    from numpy import nan
    from pandas.core.indexes.base import Index
    from pandas.core.indexes.numeric import Int64Index
    from sklearn.pipeline import Pipeline
    
    transformer_list = []
    transformer1 = DropColumns(
        drop_columns=['PROCENT_DISCOUNT', 'ID_CATEGORIE']
    )
    transformer_list.append(('drop_irrelevant_columns', transformer1))
    
    transformer2 = UniqueTargetGrainDropper(
        cv_step_size=4,
        max_horizon=4,
        n_cross_validations=3,
        target_lags=[0],
        target_rolling_window_size=0
    )
    transformer_list.append(('unique_target_grain_dropper', transformer2))
    
    transformer3 = MissingDummiesTransformer(
        numerical_columns=['SAPTAMANA_W53', 'STOC_W2', 'PRET_UNITAR', 'SAPTAMANA_W53_W104', 'SAPTAMANA_W1_W12', 'STOC_W1', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53_W56', 'STOC_M1', 'SAPTAMANA_W1', 'SAPTAMANA_W53_W64']
    )
    transformer_list.append(('make_numeric_na_dummies', transformer3))
    
    transformer4 = TimeSeriesImputer(
        end=None,
        freq='W-MON',
        impute_by_horizon=False,
        input_column=['SAPTAMANA_W53', 'STOC_W2', 'PRET_UNITAR', 'SAPTAMANA_W53_W104', 'SAPTAMANA_W1_W12', 'STOC_W1', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53_W56', 'STOC_M1', 'SAPTAMANA_W1', 'SAPTAMANA_W53_W64'],
        limit=None,
        limit_direction='forward',
        method=OrderedDict([('ffill', [])]),
        option='fillna',
        order=None,
        origin=None,
        value={'SAPTAMANA_W53': 413.0, 'STOC_W2': 118.0, 'PRET_UNITAR': 62.0, 'SAPTAMANA_W53_W104': 16456.5, 'SAPTAMANA_W1_W12': 5207.0, 'STOC_W1': 118.0, 'SAPTAMANA_W1_W4': 1738.0, 'SAPTAMANA_W1_W52': 22495.0, 'SAPTAMANA_W53_W56': 1700.0, 'STOC_M1': 120.0, 'SAPTAMANA_W1': 435.0, 'SAPTAMANA_W53_W64': 5127.0}
    )
    transformer_list.append(('impute_na_numeric_datetime', transformer4))
    
    transformer5 = ShortGrainDropper(
        cv_step_size=4,
        max_horizon=4,
        n_cross_validations=3,
        target_lags=[0],
        target_rolling_window_size=0
    )
    transformer_list.append(('grain_dropper', transformer5))
    
    transformer6 = RestoreDtypesTransformer(
        dtypes={'SAPTAMANA_W53_W64': dtype('float64'), 'STOC_W2': dtype('float64'), 'SAPTAMANA_W53': dtype('float64'), 'PRET_UNITAR': dtype('float64'), 'SAPTAMANA_W53_W104': dtype('float64'), 'SAPTAMANA_W1_W12': dtype('float64'), 'STOC_W1': dtype('float64'), 'SAPTAMANA_W1_W4': dtype('float64'), 'SAPTAMANA_W1_W52': dtype('float64'), 'SAPTAMANA_W53_W56': dtype('float64'), 'SAPTAMANA_W1': dtype('float64'), 'STOC_M1': dtype('float64'), '_automl_target_col': dtype('float64')},
        target_column='_automl_target_col'
    )
    transformer_list.append(('restore_dtypes_transform', transformer6))
    
    transformer7 = GrainIndexFeaturizer(
        categories_by_grain_cols=None,
        grain_feature_prefix='grain',
        overwrite_columns=True,
        prefix_sep='_',
        ts_frequency='W-MON'
    )
    transformer_list.append(('make_grain_features', transformer7))
    
    transformer8 = NumericalizeTransformer(
        categories_by_col={'DENUMIRE_PRODUCATOR': Index(['ENEL ENERGIE MUNTENIA SA', 'GRUP SERBAN HOLDING S.A.', 'H&M HENNES & MAURITZ SRL', 'IKEA ROMANIA S.A.',
               'LA DOI PASI S.R.L.', 'MEGA IMAGE SRL', 'PF', 'VODAFONE ROMANIA SA'],
              dtype='object'), 'grain_ID_ARTICOL': Index(['104', '105', '106', '107', '110', '116', '139', '157', '158', '159', '160', '161', '162', '163', '164', '166',
               '167', '56', '57', '58', '59', '60', '61', '62', '63', '73', '74', '75', '76', '81'],
              dtype='object'), 'ID_FIRMA': Int64Index([90, 124], dtype='int64'), 'PROC_TVA': Int64Index([0, 5, 9, 19], dtype='int64')},
        exclude_columns={'PRET_UNITAR', 'SAPTAMANA_W1_W12', 'PROCENT_DISCOUNT', 'STOC_W2', 'STOC_W1', 'SAPTAMANA_W1_W4', 'SAPTAMANA_W53_W56', 'SAPTAMANA_W53', 'SAPTAMANA_W1_W52', 'SAPTAMANA_W53_W64', 'SAPTAMANA_W1', 'STOC_M1', 'SAPTAMANA_W53_W104'},
        include_columns={'DENUMIRE_PRODUCATOR', 'PROC_TVA', 'ID_FIRMA'}
    )
    transformer_list.append(('make_categoricals_numeric', transformer8))
    
    transformer9 = TimeIndexFeaturizer(
        correlation_cutoff=0.99,
        country_or_region=None,
        datetime_columns=None,
        force_feature_list=None,
        freq='W-MON',
        holiday_end_time=None,
        holiday_start_time=None,
        overwrite_columns=True,
        prune_features=True
    )
    transformer_list.append(('make_time_index_featuers', transformer9))
    
    transformer10 = CategoryBinarizer(
        columns=[],
        drop_first=False,
        dummy_na=False,
        encode_all_categoricals=False,
        prefix=None,
        prefix_sep='_'
    )
    transformer_list.append(('make_categoricals_onehot', transformer10))
    
    pipeline = Pipeline(steps=transformer_list)
    tst = TimeSeriesTransformer(
        country_or_region=None,
        drop_column_names=['PROCENT_DISCOUNT', 'ID_CATEGORIE'],
        featurization_config=FeaturizationConfig(
            blocked_transformers=None,
            column_purposes={'ID_ARTICOL': 'Categorical', 'ID_CATEGORIE': 'Categorical', 'ID_FIRMA': 'Categorical', 'START_DATE_WEEK': 'DateTime', 'PRET_UNITAR': 'Numeric', 'DENUMIRE_PRODUCATOR': 'Categorical', 'PROC_TVA': 'Categorical', 'PROCENT_DISCOUNT': 'Numeric', 'SAPTAMANA_W1': 'Numeric', 'SAPTAMANA_W1_W4': 'Numeric', 'SAPTAMANA_W1_W12': 'Numeric', 'SAPTAMANA_W1_W52': 'Numeric', 'SAPTAMANA_W53': 'Numeric', 'SAPTAMANA_W53_W56': 'Numeric', 'SAPTAMANA_W53_W64': 'Numeric', 'SAPTAMANA_W53_W104': 'Numeric', 'STOC_W1': 'Numeric', 'STOC_W2': 'Numeric', 'STOC_M1': 'Numeric'},
            dataset_language=None,
            prediction_transform_type=None,
            transformer_params={'Imputer': []}
        ),
        force_time_index_features=None,
        freq='W-MON',
        grain_column_names=['ID_ARTICOL'],
        group=None,
        lookback_features_removed=False,
        max_horizon=4,
        origin_time_colname='origin',
        pipeline=pipeline,
        pipeline_type=TimeSeriesPipelineType.FULL,
        seasonality=4,
        time_column_name='START_DATE_WEEK',
        time_index_non_holiday_features=['_automl_year', '_automl_half', '_automl_quarter', '_automl_month', '_automl_day', '_automl_qday'],
        use_stl=None
    )
    
    return tst
    
    
def generate_preprocessor_config_0():
    '''
    Specifies a preprocessing step to be done after featurization in the final scikit-learn pipeline.
    
    Normally, this preprocessing step only consists of data standardization/normalization that is
    accomplished with sklearn.preprocessing. Automated ML only specifies a preprocessing step for
    non-ensemble classification and regression models.
    '''
    from sklearn.preprocessing import MinMaxScaler
    
    preproc = MinMaxScaler(
        clip=False,
        copy=True,
        feature_range=(0, 1)
    )
    
    return preproc
    
    
def generate_algorithm_config_0():
    from sklearn.ensemble import RandomForestRegressor
    
    algorithm = RandomForestRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_depth=None,
        max_features=0.9,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.0023646822772690063,
        min_samples_split=0.0037087774117744725,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_estimators=200,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_1():
    from sklearn.preprocessing import MinMaxScaler
    
    preproc = MinMaxScaler(
        clip=False,
        copy=True,
        feature_range=(0, 1)
    )
    
    return preproc
    
    
def generate_algorithm_config_1():
    from sklearn.ensemble import RandomForestRegressor
    
    algorithm = RandomForestRegressor(
        bootstrap=False,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_depth=None,
        max_features=0.7,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.0028629618034842247,
        min_samples_split=0.005285388593079247,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_estimators=25,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_2():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )
    
    return preproc
    
    
def generate_algorithm_config_2():
    from sklearn.ensemble import RandomForestRegressor
    
    algorithm = RandomForestRegressor(
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_depth=None,
        max_features=0.7,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.0023646822772690063,
        min_samples_split=0.00310675990983383,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_estimators=25,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_3():
    from sklearn.preprocessing import MinMaxScaler
    
    preproc = MinMaxScaler(
        clip=False,
        copy=True,
        feature_range=(0, 1)
    )
    
    return preproc
    
    
def generate_algorithm_config_3():
    from sklearn.ensemble import RandomForestRegressor
    
    algorithm = RandomForestRegressor(
        bootstrap=False,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_depth=None,
        max_features=0.7,
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.004196633747563344,
        min_samples_split=0.008991789964660124,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_estimators=25,
        n_jobs=-1,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_preprocessor_config_4():
    from sklearn.preprocessing import RobustScaler
    
    preproc = RobustScaler(
        copy=True,
        quantile_range=(10, 90),
        unit_variance=False,
        with_centering=False,
        with_scaling=False
    )
    
    return preproc
    
    
def generate_algorithm_config_4():
    from sklearn.tree import DecisionTreeRegressor
    
    algorithm = DecisionTreeRegressor(
        ccp_alpha=0.0,
        criterion='friedman_mse',
        max_depth=None,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.011942468634416342,
        min_samples_split=0.052853885930792446,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        random_state=None,
        splitter='best'
    )
    
    return algorithm
    
    
def generate_preprocessor_config_5():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    
    return preproc
    
    
def generate_algorithm_config_5():
    from xgboost.sklearn import XGBRegressor
    
    algorithm = XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=1,
        enable_categorical=False,
        gamma=0,
        gpu_id=-1,
        importance_type=None,
        interaction_constraints='',
        learning_rate=0.300000012,
        max_delta_step=0,
        max_depth=6,
        min_child_weight=1,
        missing=numpy.nan,
        monotone_constraints='()',
        n_estimators=100,
        n_jobs=0,
        num_parallel_tree=1,
        objective='reg:squarederror',
        predictor='auto',
        random_state=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=1,
        tree_method='auto',
        validate_parameters=1,
        verbose=-10,
        verbosity=0
    )
    
    return algorithm
    
    
def generate_algorithm_config():
    '''
    Specifies the actual algorithm and hyperparameters for training the model.
    
    It is the last stage of the final scikit-learn pipeline. For ensemble models, generate_preprocessor_config_N()
    (if needed) and generate_algorithm_config_N() are defined for each learner in the ensemble model,
    where N represents the placement of each learner in the ensemble model's list. For stack ensemble
    models, the meta learner generate_algorithm_config_meta() is defined.
    '''
    from azureml.training.tabular.models.voting_ensemble import PreFittedSoftVotingRegressor
    from sklearn.pipeline import Pipeline
    
    pipeline_0 = Pipeline(steps=[('preproc', generate_preprocessor_config_0()), ('model', generate_algorithm_config_0())])
    pipeline_1 = Pipeline(steps=[('preproc', generate_preprocessor_config_1()), ('model', generate_algorithm_config_1())])
    pipeline_2 = Pipeline(steps=[('preproc', generate_preprocessor_config_2()), ('model', generate_algorithm_config_2())])
    pipeline_3 = Pipeline(steps=[('preproc', generate_preprocessor_config_3()), ('model', generate_algorithm_config_3())])
    pipeline_4 = Pipeline(steps=[('preproc', generate_preprocessor_config_4()), ('model', generate_algorithm_config_4())])
    pipeline_5 = Pipeline(steps=[('preproc', generate_preprocessor_config_5()), ('model', generate_algorithm_config_5())])
    algorithm = PreFittedSoftVotingRegressor(
        estimators=[
            ('model_0', pipeline_0),
            ('model_1', pipeline_1),
            ('model_2', pipeline_2),
            ('model_3', pipeline_3),
            ('model_4', pipeline_4),
            ('model_5', pipeline_5),
        ],
        weights=[0.6666666666666666, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667]
    )
    
    return algorithm
    
    
def build_model_pipeline():
    '''
    Defines the scikit-learn pipeline steps.
    
    For time-series forecasting models, the scikit-learn pipeline is wrapped in a ForecastingPipelineWrapper,
    which has some additional logic needed to properly handle time-series data depending on the applied algorithm.
    '''
    from azureml.training.tabular.models.forecasting_pipeline_wrapper import ForecastingPipelineWrapper
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('tst', generate_data_transformation_config()),
            ('model', generate_algorithm_config())
        ]
    )
    forecast_pipeline_wrapper = ForecastingPipelineWrapper(pipeline, stddev=[19.474354916082916])
    
    return forecast_pipeline_wrapper


def train_model(X, y, sample_weights=None, transformer=None):
    '''
    Calls the fit() method to train the model.
    
    The return value is the model fitted/trained on the input data.
    '''
    
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    '''
    Calculates the metrics that can be used to evaluate the model's performance.
    
    Metrics calculated vary depending on the experiment type. Classification, regression and time-series
    forecasting jobs each have their own set of metrics that are calculated.'''
    
    from azureml.training.tabular.preprocessing._dataset_binning import get_dataset_bins
    from azureml.training.tabular.score.scoring import score_forecasting
    from azureml.training.tabular.score.scoring import score_regression
    
    y_pred, _ = model.forecast(X_test)
    y_min = np.min(y)
    y_max = np.max(y)
    y_std = np.std(y)
    
    bin_info = get_dataset_bins(cv_splits, X, None, y)
    regression_metrics_names, forecasting_metrics_names = get_metrics_names()
    metrics = score_regression(
        y_test, y_pred, regression_metrics_names, y_max, y_min, y_std, sample_weights, bin_info)
    
    try:
        horizons = X_test['horizon_origin'].values
    except Exception:
        # If no horizon is present we are doing a basic forecast.
        # The model's error estimation will be based on the overall
        # stddev of the errors, multiplied by a factor of the horizon.
        horizons = np.repeat(None, y_pred.shape[0])
    
    featurization_step = generate_data_transformation_config()
    grain_column_names = featurization_step.grain_column_names
    time_column_name = featurization_step.time_column_name
    
    forecasting_metrics = score_forecasting(
        y_test, y_pred, forecasting_metrics_names, horizons, y_max, y_min, y_std, sample_weights, bin_info,
        X_test, X, y, grain_column_names, time_column_name)
    metrics.update(forecasting_metrics)
    return metrics


def get_metrics_names():
    
    regression_metrics_names = [
        'root_mean_squared_error',
        'root_mean_squared_log_error',
        'spearman_correlation',
        'r2_score',
        'residuals',
        'mean_absolute_error',
        'explained_variance',
        'predicted_true',
        'median_absolute_error',
        'mean_absolute_percentage_error',
    ]
    forecasting_metrics_names = [
        'forecast_adjustment_residuals',
        'forecast_mean_absolute_percentage_error',
        'forecast_table',
        'forecast_residuals',
    ]
    return regression_metrics_names, forecasting_metrics_names


def get_metrics_log_methods():
    
    metrics_log_methods = {
        'root_mean_squared_error': 'log',
        'root_mean_squared_log_error': 'log',
        'spearman_correlation': 'log',
        'r2_score': 'log',
        'residuals': 'log_residuals',
        'mean_absolute_error': 'log',
        'explained_variance': 'log',
        'forecast_mean_absolute_percentage_error': 'Skip',
        'forecast_table': 'Skip',
        'forecast_adjustment_residuals': 'Skip',
        'predicted_true': 'log_predictions',
        'median_absolute_error': 'log',
        'mean_absolute_percentage_error': 'log',
        'forecast_residuals': 'Skip',
    }
    return metrics_log_methods


def main(training_dataset_uri=None):
    '''
    Runs all functions defined above.
    '''
    
    from azureml.automl.core.inference import inference
    from azureml.core.run import Run
    from azureml.training.tabular.score._cv_splits import _CVSplits
    from azureml.training.tabular.score.scoring import aggregate_scores
    
    import mlflow
    
    # The following code is for when running this code as part of an AzureML script run.
    run = Run.get_context()
    
    df = get_training_dataset(training_dataset_uri)
    X, y, sample_weights = prepare_data(df)
    tst = generate_data_transformation_config()
    tst.fit(X, y)
    ts_param_dict = tst.parameters
    short_series_dropper = next((step for key, step in tst.pipeline.steps if key == 'grain_dropper'), None)
    if short_series_dropper is not None and short_series_dropper.has_short_grains_in_train and grains is not None and len(grains) > 0:
        # Preprocess X so that it will not contain the short grains.
        dfs = []
        X['_automl_target_col'] = y
        for grain, df in X.groupby(grains):
            if grain in short_series_processor.grains_to_keep:
                dfs.append(df)
        X = pd.concat(dfs)
        y = X.pop('_automl_target_col').values
        del dfs
    cv_splits = _CVSplits(X, y, frac_valid=None, CV=3, n_step=4, is_time_series=True, task='regression', timeseries_param_dict=ts_param_dict)
    scores = []
    for X_train, y_train, sample_weights_train, X_valid, y_valid, sample_weights_valid in cv_splits.apply_CV_splits(X, y, sample_weights):
        partially_fitted_model = train_model(X_train, y_train, transformer=tst)
        metrics = calculate_metrics(partially_fitted_model, X, y, sample_weights, X_test=X_valid, y_test=y_valid, cv_splits=cv_splits)
        scores.append(metrics)
        print(metrics)
    model = train_model(X_train, y_train, transformer=tst)
    
    metrics = aggregate_scores(scores)
    metrics_log_methods = get_metrics_log_methods()
    print(metrics)
    for metric in metrics:
        if metrics_log_methods[metric] == 'None':
            logger.warning("Unsupported non-scalar metric {}. Will not log.".format(metric))
        elif metrics_log_methods[metric] == 'Skip':
            pass # Forecasting non-scalar metrics and unsupported classification metrics are not logged
        else:
            getattr(run, metrics_log_methods[metric])(metric, metrics[metric])
    cd = inference.get_conda_deps_as_dict(True)
    
    # Saving ML model to outputs/.
    signature = mlflow.models.signature.infer_signature(X, y)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='outputs/',
        conda_env=cd,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    
    run.upload_folder('outputs/', 'outputs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_uri', type=str, default='azureml://locations/westeurope/workspaces/def84eb2-d667-4eb4-8f9a-70505c89c4e3/data/weekly_product_sale_train/versions/1',     help='Default training dataset uri is populated from the parent run')
    args = parser.parse_args()
    
    try:
        main(args.training_dataset_uri)
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise