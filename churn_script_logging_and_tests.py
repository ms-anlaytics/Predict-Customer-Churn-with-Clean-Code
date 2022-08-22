'''
Unit test script to test and log the performance of churn_library.py scripts

Author: Michael Stephenson
Creation Data: 8/21/2022
'''

import logging
from churn_library import import_data, perform_eda, perform_feature_engineering, \
    encoder_helper, train_new_models


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - tests import of data file
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing import_eda: The data file found in 'test_import' call ")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The data file missing rows and columns in 'test_import' call")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    try:
        dataframe = import_data("./data/bank_data.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        logging.info('Testing EDA data:SUCCESS')

    except FileNotFoundError as err:
        logging.error(
            "Testing perform_eda: EDA Images not saved in 'test_eda' call")
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The data file missing rows and columns in 'test_eda' call")
        raise err
    try:
        assert categorical_columns
        assert input_data
        assert output_data
        logging.info(
            'SUCCESS: perform_eda variables !Null in "test_eda" call ')
    except BaseException:
        logging.error(
            'ERROR: perform_eda returned Null variable in "test_eda" call ')


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        input_data, categorical_columns, output_data = perform_eda(dataframe)
        encoded_dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], 'Churn')

        logging.info(
            'SUCCESS:New categorical mean dataframe created in "test_encoder_helper" call ')
    except BaseException:
        logging.error(
            'ERROR: creating categorical mean dataframe failed in "test_encoder_helper" call')
    try:
        assert encoded_dataframe.shape[0] > 0
        assert encoded_dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "test_encoder_helper: data file missing rows and columns")
        raise err
    try:
        assert input_data
        assert output_data
        logging.info(
            'SUCCESS: perform_eda variables !Null in "test_encoder_helper" call ')
    except BaseException:
        logging.error(
            'ERROR: perform_eda returned Null variable in "test_encoder_helper" call ')


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        dependent = 'Churn'
        dataframe = import_data("./data/bank_data.csv")
        x_data, categorical_columns, x_data = perform_eda(dataframe)

        dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], dependent)

        x_training_data, x_testing_data, y_training_data, \
            y_testing_data = perform_feature_engineering(dataframe, dependent)

        logging.info(
            'SUCCESS:Data split into training and testing in "perform_feature_engineering" call')
    except BaseException:
        logging.error(
            'ERROR:Problem splitting into training and testing in "perform_feature_engineering" call')
    try:
        if not x_training_data.shape[0] == y_training_data.shape[0]:
            raise AssertionError(x_training_data[0], y_training_data.shape[0])
    except AssertionError as err:
        logging.error(
            'X and y are not the same shape in "perform_feature_engineering" call')
        raise err
    try:
        # To test that a dataframe is not empty
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info(
            'SUCCESS: dataframe is not empty in "test_perform_feature_engineering" call ')
    except BaseException:
        logging.error(
            'ERROR: dataframe is "NULL" in "test_perform_feature_engineering" call ')
        raise err
    try:
        assert x_testing_data
        assert y_testing_data
        assert x_data
        logging.info(
            'SUCCESS: perform_feature_engineering !Null in "test_perform_feature_engineering" call ')
    except BaseException:
        logging.error(
            'ERROR: perform_feature_engineering returned Null variable in "test_perform_feature_engineering" call ')


def test_train_models(train_new_models):
    '''
    test train_models
    '''

    try:
        dependent = 'Churn'
        dataframe = import_data("./data/bank_data.csv")
        x_data, categorical_columns, y_data = perform_eda(dataframe)

        dataframe = encoder_helper(
            dataframe, dataframe[categorical_columns], dependent)

        x_training_data, x_testing_data, y_training_data, y_testing_data\
            = perform_feature_engineering(dataframe, dependent)

        lrc, cv_rfc, y_training_data, y_testing_data, y_train_preds_lr, y_train_preds_rf,\
            y_test_preds_lr, y_test_preds_rf = \
            train_new_models(x_training_data, x_testing_data, y_training_data, y_testing_data)

        cv_rfc.fit(x_training_data, y_training_data)
        lrc.fit(x_training_data, y_training_data)

        logging.info('SUCESS:Models trained in "train_models" call')
    except BaseException:
        logging.error('ERROR: failed to train models in "train_models" call ')

    try:
        assert y_train_preds_rf
        assert y_test_preds_rf
        assert y_train_preds_lr
        assert y_test_preds_lr
        assert x_data
        assert y_data

        logging.info(
            'SUCCESS: train_new_models variables !Null in "test_train_models" call ')
    except BaseException:
        logging.error(
            'ERROR: train_new_models returned a Null variable in \
                "test_train_models" call')

    try:
        if not x_training_data.shape[0] == y_training_data.shape[0]:
            raise AssertionError(
                (x_training_data.shape[0],
                 y_training_data.shape[0]))

        if not x_testing_data.shape[0] == y_testing_data.shape[0]:
            raise AssertionError(
                (x_training_data.shape[0], y_testing_data.shape[0]))
    except AssertionError as err:
        logging.error(
            'X and y are not the same shape in "test_train_models" call')
        raise err


if __name__ == "__main__":

    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_new_models)
