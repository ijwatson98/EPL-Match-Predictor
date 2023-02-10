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
    
    
    def __init__(self, data):
        
        self.data = pd.read_csv(data, index_col=0)
        self._predictions = {}
        self._performance = {}
        self.model = None
        
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
        
                  
    
    def optimise(self, objective, space, max_evals=1000):
        
        self.trials = Trials()
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
        
        self.model = best_trial_obj['result']['model']
    
    
    def fit(self):
        
        evaluation=[(self.X_train, self.y_train), (self.X_test, self.y_test)]
        self.model.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)
        
        
    def predict(self, model_name):
        
        self.pred = self.model.predict(self.X_test)
        self._predictions[model_name] = self.pred
        
        
    def validate(self, model_name):
        
        self._performance["Accuracy - "+model_name] = accuracy_score(self.y_test, self._predictions[model_name])
        self._performance["F1 - "+model_name] = self.macro_f1(self.y_test, self._predictions[model_name])
        

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
        
        filter = (self.trials_df[constant]==constant_val)

        fig = go.Figure(
            data=go.Contour(
                z=self.trials_df[filter]["loss"],
                x=self.trials_df[filter][parameter1],
                y=self.trials_df[filter][parameter2],
                contours=dict(
                    showlabels=True,  # show labels on contours
                    labelfont=dict(size=12, color="white",),  # label font properties
                ),
                colorbar=dict(title="loss", titleside="right",),
                hovertemplate="loss: %{z}<br>"+parameter1+": %{x}<br>"+parameter2+": %{y}<extra></extra>",
            )
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
        plt.plot(self.results["validation_0"]["merror"], label="Training loss", c="darkturquoise")
        plt.plot(self.results["validation_1"]["merror"], label="Validation loss", c="darkblue")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        

    @staticmethod
    def true_positive(y_true, y_pred):
        
        tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
                
        return tp
    
    @staticmethod
    def true_negative(y_true, y_pred):
        
        tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 0:
                tn += 1
                
        return tn
    @staticmethod
    def false_positive(y_true, y_pred):
        
        fp = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 0 and yp == 1:
                fp += 1
                
        return fp
    @staticmethod
    def false_negative(y_true, y_pred):
        
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 0:
                fn += 1
                
        return fn 
    

    def macro_f1(self, y_true, y_pred):
    
        # find the number of classes
        num_classes = len(np.unique(y_true))
    
        # initialize f1 to 0
        f1 = 0
        
        # loop over all classes
        for class_ in list(np.unique(y_true)):
            
            # all classes except current are considered negative
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            
            # compute true positive for current class
            tp = MatchPredictorML.true_positive(temp_true, temp_pred)
            
            # compute false negative for current class
            fn = MatchPredictorML.false_negative(temp_true, temp_pred)
            
            # compute false positive for current class
            fp = MatchPredictorML.false_positive(temp_true, temp_pred)
            
            # compute recall for current class
            temp_recall = tp / (tp + fn + 1e-6)
            
            # compute precision for current class
            temp_precision = tp / (tp + fp + 1e-6)
            
            
            temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
            
            # keep adding f1 score for all classes
            f1 += temp_f1
            
        # calculate and return average f1 score over all classes
        f1 /= num_classes
        
        return f1
        