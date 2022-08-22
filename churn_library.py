'''
churn_library.py scripts refactored from churn_library.ipynb.

Author: Michael Stephenson
Creation Data: 8/21/2022
'''


# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# cd C:\Users\mlang\OneDrive\DebOpsMLEngineer\project_1


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
            r"./data/bank_data.csv"
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except BaseException:
        pass


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # plots Chrun to jpg file in ./images
        plt.figure(figsize=(10, 5))
        df['Churn'].hist()
        plt.title('Churn')
        plt.savefig(
            './images/eda/Churn_plot.jpg',
            bbox_inches='tight',
            dpi=800)

        # plots Customer_Age to jpg in ./images
        plt.figure(figsize=(10, 5))
        df['Customer_Age'].hist()
        plt.title('Customer_Age')
        plt.savefig(
            './images/eda/Customer_Age.jpg',
            bbox_inches='tight',
            dpi=800)

        # plots Marital_Status to jpg in ./images
        plt.figure(figsize=(10, 5))
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.title('Marital_Status')
        plt.savefig(
            './images/eda/Marital_Status.jpg',
            bbox_inches='tight',
            dpi=800)

        # plots Total_Trans_Ct' to jpg in ./images
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Total_Trans_Ct')
        plt.savefig(
            './images/eda/Total_Trans_Ct.jpg',
            bbox_inches='tight',
            dpi=800)

        # plots Correlation Table' to jpg in ./images
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Correlation Table')
        plt.savefig(
            './images/eda/Correlation.jpg',
            bbox_inches='tight',
            dpi=800)
        # plt.show()

        X = pd.DataFrame()

        y = df['Churn']

        return X, cat_columns, y
    except BaseException:
        pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for categorical data
    '''

    try:
        group_lst = []
        name = ''
        for i in category_lst:

            groups = df.groupby(i).mean()[response]
            name = str(i + '_' + response)
            for val in df[i]:
                group_lst.append(groups.loc[val])
            df[name] = group_lst
            group_lst.clear()

        return df
    except BaseException:
        pass


def perform_feature_engineering(df, responce):
    '''
    input:
            df: pandas dataframe

            responce: String for index y column

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    try:
        y = df[responce]
        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']

        X = df[keep_cols]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test
    except BaseException:
        pass


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    try:

        plt.rc('figure', figsize=(5, 5))

        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.savefig(
            "'./images/results/rf_train_results.jpeg",
            bbox_inches='tight',
            dpi=1000)

        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.savefig(
            "'./images/results/rf_test_results.jpeg",
            bbox_inches='tight',
            dpi=1000)

        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_train, y_train_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.savefig(
            "./images/results/logistic_train_results.jpeg",
            bbox_inches='tight',
            dpi=1000)

        plt.text(0.01, 1.25, str('Logistic Regression Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.savefig(
            "./images/results/logistic_test_results.jpeg",
            bbox_inches='tight',
            dpi=1000)

        plt.axis('off')
    except BaseException:
        pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        explainer = shap.TreeExplainer(model.best_estimator_)
        shap_values = explainer.shap_values(X_data)
        shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
        plt.savefig(
            "./images/results/shap_feature_importance.png",
            bbox_inches='tight',
            dpi=1000)

        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(
            "./images/results/RFfeature_importance.png",
            bbox_inches='tight',
            dpi=1000)

    except BaseException:
        pass


def train_new_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    try:
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference:
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='liblinear', max_iter=3000)
        # solver='lbfgs'

        param_grid = {
            'n_estimators': [
                200, 500], 'max_features': [
                'auto', 'sqrt'], 'max_depth': [
                4, 5, 100], 'criterion': [
                    'gini', 'entropy']}

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            cv_rfc.best_estimator_,
            X_test,
            y_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig("./images/results/roc_curve_results.jpeg",
                    bbox_inches='tight',
                    dpi=1000)

        joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
        joblib.dump(lrc, "./models/logistic_model.pkl")

        #feature_importance_plot(model, X_data, output_pth)

        return lrc, cv_rfc, y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
    except BaseException:
        print('err')
        pass
