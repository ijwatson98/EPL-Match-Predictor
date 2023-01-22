# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:08:47 2023

@author: Surface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from hyperopt import Trials, fmin, tpe, STATUS_OK
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from plotly import graph_objects as go

class MatchPredictorML():
    
    
    def __init__(self):
        
        self.data = pd.read_csv("matches.csv", index_col=0)
        self.trials = Trials()
        self._predictions = {}
        self._performance = {}
        
        pass
    
    
    def corrs(self, final_df, target):
        
        sns.heatmap(final_df.corr()[[target]].sort_values(target, ascending=False), annot=True, cmap="mako", vmax=1, vmin=-1)
        plt.show()
     
        
    def split(self, final_df, target):
        
        train, test = np.split(final_df, [int(.8*len(final_df))])
        train, val = np.split(train, [int(.8*len(train))])
        
        self.X_train = train.drop(target, axis=1)
        self.y_train = train[target]

        self.X_test = test.drop(target, axis=1)
        self.y_test = test[target]
        
        self.X_val = val.drop(target, axis=1)
        self.y_val = val[target]
          
        
    def objective(self, space):
    
        model = xgb.XGBClassifier(objective="multi:softmax", 
                                  num_class=3,
                                  max_depth=space['max_depth'],
                                  min_child_weight=space['min_child_weight'],
                                  n_estimators=space['n_estimators'],
                                  gamma=space['gamma'],
                                  learning_rate=space['learning_rate'],
                                  reg_lambda=space['reg_lambda'],
                                  eval_metric="mlogloss",
                                  early_stopping_rounds=space['early_stopping_rounds'],
                                  subsample=space['subsample']
                                 )
        
        evaluation=[(self.X_test, self.y_test)]
        
        model.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)
        
        y_pred_probs = model.predict_proba(self.X_test)
        y_pred = model.predict(self.X_test)
        
        score1 = cohen_kappa_score(self.y_test, y_pred)
        score2 = roc_auc_score(self.y_test, y_pred_probs, multi_class='ovo')
        
        loss = 1-(0.7*score1+0.3*score2)/2
        
        return {'loss': loss, 'status': STATUS_OK, 'model': model}
    
    
    def optimise(self, objective, space, max_evals=1000):
        
        self.best_params = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                trials=self.trials,
                                max_evals=max_evals)
        
        
    def getBestModelfromTrials(self):
            
        valid_trial_list = [trial for trial in self.trials
                                if STATUS_OK == trial['result']['status']]
        losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
        index_having_minumum_loss = np.argmin(losses)
        best_trial_obj = valid_trial_list[index_having_minumum_loss]
        
        return best_trial_obj['result']['model']
    
    
    def fit(self):
        
        self.model = self.getBestModelfromTrials()
        
        evaluation=[(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)
        
        
    def predict(self, model_name):
        
        self.pred = self.model.predict(self.X_test)
        self._predictions[model_name] = self.pred
        
        
    def validate(self, model_name):
        
        self._performance[model_name] = accuracy_score(self._predictions[model_name], self.y_test)
        

    def confusionMatrix(self):
        
        self.conf_matrix = confusion_matrix(self.y_test, self.pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=self.conf_matrix)
        cm_display.plot()
        plt.show()
        
        
    def contour2D(self, parameter1, parameter2, constant, constant_val):
        
        def unpack(x):
            if x:
                return x[0]
            return np.nan
        
        self.trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in self.trials])
        
        self.trials_df["loss"] = [t["result"]["loss"] for t in self.trials]
        self.trials_df["trial_number"] = self.trials_df.index
        
        filter = (self.trials_df[constant] == constant_val)

        fig = go.Figure(
            data=[go.Contour(
                z=self.trials_df[filter]["loss"],
                x=self.trials_df[filter][parameter1],
                y=self.trials_df[filter][parameter2],
                contours=dict(
                    showlabels=True,  # show labels on contours
                    labelfont=dict(size=12, color="white",),  # label font properties
                ),
                colorbar=dict(title="loss", titleside="right",),
                hovertemplate="loss: %{z}<br>"+parameter1+": %{x}<br>"+parameter2+": %{y}<extra></extra>",
            )]
        )
    
        fig.update_layout(
            xaxis_title=parameter1,
            yaxis_title=parameter2,
            title={
                "text": parameter1 + " vs. " + parameter2 + " | " + constant + " == " + str(constant_val),
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
            },
        )
        
        return fig
        
    
    def featureImportance(self):
        
        sort = self.model.feature_importances_.argsort()
        plt.barh(self.X_train.columns[sort], self.model.feature_importances_[sort])
        plt.xlabel("Feature Importance")
        plt.show()
        
            
    def learningCurve(self):
        
        self.results = self.model.evals_result()

        plt.figure(figsize=(10,7))
        plt.plot(self.results["validation_0"]["mlogloss"], label="Training loss")
        plt.plot(self.results["validation_1"]["mlogloss"], label="Validation loss")
        plt.axvline(self.model.best_ntree_limit, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()