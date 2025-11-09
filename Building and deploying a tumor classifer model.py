# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:48:18 2025

@author: Asus
"""

#%%
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
#%%
pd.set_option('display.max_columns', None)
df = "C:\\Users\\Asus\\Desktop\\Cancer_Data.csv"
df = pd.read_csv(df)
#%%
dfnull = df['Unnamed: 32'].isna().sum()
#%%
df = df.drop(['id', 'Unnamed: 32'], axis=1)
#%%
C1 = (df["diagnosis"] == 'M').sum()
C2 = (df["diagnosis"] == 'B').sum()
#%%
df['diagnosis'] = df['diagnosis'] .replace({'B': 1, 'M': 0})
#%%
df_train, df_test = sklearn.model_selection.train_test_split(df,test_size=0.15)
print("df:", df.shape)
print("df_train:", df_train.shape)
print("df_test:", df_test.shape)
#%%
dfnull = df_train.isna().sum()
#%%
dfduplicate =df_train.duplicated().sum()
#%%
minval = df_train.min()
maxval = df_train.max()
#%%
Q1 = df_train['area_mean'].quantile(0.10)
Q3 = df_train['area_mean'].quantile(0.90)
IQR = Q3 - Q1

df_train = df_train[(df_train['area_mean'] >= Q1 - 1.5 * IQR) & (df_train['area_mean'] <= Q3 + 1.5 * IQR)]
#%%
Q1 = df_train['area_se'].quantile(0.10)
Q3 = df_train['area_se'].quantile(0.90)
IQR = Q3 - Q1

df_train = df_train[(df_train['area_se'] >= Q1 - 1.5 * IQR) & (df_train['area_se'] <= Q3 + 1.5 * IQR)]
#%%
Q1 = df_train['area_worst'].quantile(0.10)
Q3 = df_train['area_worst'].quantile(0.90)
IQR = Q3 - Q1

df_train = df_train[(df_train['area_worst'] >= Q1 - 1.5 * IQR) & (df_train['area_worst'] <= Q3 + 1.5 * IQR)]
#%%
df_train.shape
#%%
x_train = df_train.drop(["diagnosis"], axis = 1)
y_train = df_train["diagnosis"]

x_test = df_test.drop(["diagnosis"], axis = 1)
y_test = df_test["diagnosis"]

print("x train size:", x_train.shape)
print("x test size:", x_test.shape)
print("y train size:", y_train.shape)
print("y test size:", y_test.shape)
#%%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
#%%
RFC_parameters_grid = {"criterion" : ["gini", "entropy", "log_loss"],
                   "n_estimators": ( 50, 100),
                        "max_depth": (5,10,None)}
RFC_model = sklearn.model_selection.GridSearchCV(sklearn.ensemble.RandomForestClassifier(),
             RFC_parameters_grid, scoring="accuracy")
RFC_model.fit(x_train, y_train)
print ("accuracy {:.2f}".format(RFC_model.best_score_))
print ("best found hyperparameters  = {}".format(RFC_model.best_params_))
#%%
ada_parameters_grid = { "learning_rate" : [ 0.1, 0.2, 1.0],
                   "n_estimators": (20, 50),
                   "algorithm": ('SAMME', 'SAMME.R')}
ADAmodel = sklearn.model_selection.GridSearchCV(sklearn.ensemble.AdaBoostClassifier(),
             ada_parameters_grid, scoring="accuracy")
ADAmodel.fit(x_train, y_train)
print ("accuracy {:.2f}".format(ADAmodel.best_score_))
print ("best found hyperparameters = {}".format(ADAmodel.best_params_))
#%%
ETC_parameters_grid = { "n_estimators": [50,100],
                       "criterion": ['gini', 'entropy'],
                        "max_depth": [None,5,10],
                       "min_impurity_decrease":[0.0, 0.5],
                        "bootstrap": [True],
                   "oob_score": (True, False)}
ETCmodel = sklearn.model_selection.GridSearchCV(sklearn.ensemble.ExtraTreesClassifier(),
             ETC_parameters_grid, scoring="accuracy")
ETCmodel.fit(x_train, y_train)
print ("accuracy {:.2f}".format(ETCmodel.best_score_))
print ("best found hyperparameters = {}".format(ETCmodel.best_params_))
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Define classifiers
T = DecisionTreeClassifier()
# setting probability as true for SVC will make it take into consideration the probality of the results of the classifcation
S = SVC(probability=True)
MNB = MultinomialNB()
# make a big border for max iteraitons to remove limits and any wanrings and to make reuslts as accurate as possible
L = LogisticRegression(max_iter=10000)

# Define parameter grids for each classifier
T_param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10],
                'min_samples_split': [1, 2], 'min_impurity_decrease': [1, 2, 0.0]}
S_param_grid = {'kernel': ['linear', 'rbf', 'poly']}
L_param_grid = {'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [500, 1000, 2000]}
MNB_param_grid = {'fit_prior': [True, False]}
#%%
estimators = [('dt', T), ('svm', S), ('NB', MNB), ('LR', L)]

# Define parameter grid for Voting Classifier
VC_parameters_grid = {"voting": ['hard', 'soft']}

# Create Voting Classifier
voting_classifier = VotingClassifier(estimators)

# Perform Grid Search with cross-validation
VC_model = GridSearchCV(voting_classifier, VC_parameters_grid, scoring="accuracy")
VC_model.fit(x_train, y_train)

# Print the best accuracy and hyperparameters
print("Accuracy: {:.2f}".format(VC_model.best_score_))
print("Best hyperparameters:", VC_model.best_params_)
#%%
Bestmodel =  RFC_model.best_estimator_ if RFC_model.best_score_ > ADAmodel.best_score_ and RFC_model.best_score_ > ETCmodel.best_score_ and RFC_model.best_score_ > VC_model.best_score_ else ADAmodel.best_estimator_ if ADAmodel.best_score_ > ETCmodel.best_score_ and ADAmodel.best_score_ > VC_model.best_score_ else ETCmodel.best_estimator_ if ETCmodel.best_score_ > VC_model.best_score_ else VC_model.best_estimator_
Bestmodel.fit(x_train, y_train)
#%%
y_pred = Bestmodel.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
CM = confusion_matrix(y_test, y_pred)

print("accuracy:", accuracy)
print("confusion matrix:\n", CM)
print("FI:", f1)
print("Precision:", precision)
print("recall:", recall)
#%%
# importing needed library and saving the model
import joblib

joblib.dump(Bestmodel, "the_model.pkl")
print("\nModel saved as the_model.pkl")
#%%
# example usage of the model
load_model = joblib.load("the_model.pkl")

example = x_test.iloc[[5]]
prediction = load_model.predict(example)
print(prediction)
#%%
# deploying the model on swagger
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)

# min swagger configuration
swagger_template = {
    "swagger": "2.0",
    "info":{
        "the title": "Classifying Tumor API",
        "description": "predict if the tumor is malignent or benign",
        "version": "1.0.0"
        }
    }

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
            }
        ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
    }

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# loading the MLM model
model = joblib.load("the_model.pkl")

@app.route('/')
def home():
    return "Tumor classification model API is running, can test it with swagger UI"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict tumor type (benign = 0, malignent = 1)
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
     - in: body
       name: input
       required: true
       schema:
         type: object
         properties:
            radius_mean:
              type: number
              example: 14
            texture_mean:
              type: number
              example: 20
            perimeter_mean:
              type: number
              example: 1
            area_mean:
              type: number
              example: 1
            smoothness_mean:
              type: number
              example: 0.2
            compactness_mean:
              type: number
              example: 1
            concavity_mean:
              type: number
              example: 1
            concave_points_mean:
              type: number
              example: 1
            symmetry_mean:
              type: number
              example: 1
            fractal_dimension_mean:
              type: number
              example: 1
            radius_se:
              type: number
              example: 1
            texture_se:
              type: number
              example: 1
            perimeter_se:
              type: number
              example: 1
            area_se:
              type: number
              example: 1
            smoothness_se:
              type: number
              example: 1
            compactness_se:
              type: number
              example: 1
            concavity_se:
              type: number
              example: 1
            concave_points_se:
              type: number
              example: 1
            symmetry_se:
              type: number
              example: 1
            fractal_dimension_se:
              type: number
              example: 1
            radius_worst:
              type: number
              example: 1
            texture_worst:
              type: number
              example: 1
            perimeter_worst:
              type: number
              example: 1
            area_worst:
              type: number
              example: 14
            smoothness_worst:
              type: number
              example: 1
            compactness_worst:
              type: number
              example: 1
            concavity_worst:
              type: number
              example: 1
            concave_points_worst:
              type: number
              example: 1
            symmetry_worst:
              type: number
              example: 1
            fractal_dimension_worst:
              type: number
              example: 1
    responses:
      200:
        description: The model result
        schema:
          type: object
          properties:
            prediction:
              type: integer
              example: 1
    """
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'Diagnosis': int(prediction)})
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
 #%%
#Dataset refrencing:
#aha, E. (2023) Cancer data, Kaggle. Available at: https://www.kaggle.com/datasets/erdemtaha/cancer-data (Accessed: 27 June 2023).