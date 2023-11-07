import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_curve, roc_auc_score
import pickle

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB


def model_building(X, y, test, model, params = None, k = 1) :
    
    if params == None :
        model.fit(X, y)
        
        # return fitted model & train-test predictions
        return (model, model.predict(X), model.predict(test))
    
    else :
        model_cv = GridSearchCV(model, param_grid = params, cv = k)
        model_cv.fit(X, y)
        model = model_cv
        
        # return and extra object for all cross validation operations
        return (model_cv, model, model.predict(X), model.predict(test))
    
def model_evaluation(y_train, pred_train, y_test, pred_test) :
    
    print('''
            +--------------------------------------+
            | CLASSIFICATION REPORT FOR TRAIN DATA |
            +--------------------------------------+''')
    print(classification_report(y_train, pred_train))
    print(confusion_matrix(y_train, pred_train))
    print("F1 Score: ",f1_score(y_train, pred_train,average="macro"))
    
    print('''
            +--------------------------------------+
            | CLASSIFICATION REPORT FOR TEST DATA  |
            +--------------------------------------+''')
    print(classification_report(y_test, pred_test))
    print(confusion_matrix(y_test, pred_test))
    print("F1 Score: ",f1_score(y_test, pred_test,average="macro"))


data_raw = pd.read_csv("data-problem-statement-1-heart-disease.csv")


data_processed = data_raw.copy()

cat_columns =["sex","cp","fbs","restecg","exang","slope","ca","thal"]
for col_name in cat_columns:
    data_processed[col_name]= data_processed[col_name].astype('category')

X = data_processed.drop("condition",axis=1)
y = data_processed["condition"]

scaler = MinMaxScaler()

# transform training data
col_list = ["trestbps","chol","thalach","oldpeak"]
for col in col_list:
    X[col] = scaler.fit_transform(X[col].to_numpy().reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,random_state=111,stratify=y)

# model_list = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(),RandomForestClassifier(),SVC()]
model_list =[GaussianNB()]
for model in model_list:
    print("\n***** Performing for {0} *****".format(model))
    model, pred_train, pred_test = model_building(X_train, y_train,X_test,model,params = None)
    try:
        plot_roc_curve(model,X_test,y_test)
    except:
        print("Unable to create a model for {0}".format(model))
    model_evaluation(y_train, pred_train, y_test, pred_test)

pickle.dump(model, open('finalmodel.pkl','wb'))

print("Model file executed successfully")