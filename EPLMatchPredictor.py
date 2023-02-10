
import sys
sys.path.append(r"C:\Users\Surface\OneDrive\Documents\Python Projects\EPL_Match_Predictor")

import pandas as pd
from MatchPredictorMLClass import MatchPredictorML

ML = MatchPredictorML("C:/Users/Surface/OneDrive/Documents/Python Projects/EPL_Match_Predictor/matches.csv")
matches = ML.data

# =============================================================================
# Data Cleaning and Feature Engineering
# =============================================================================

matches.head()

matches.columns

matches.drop(["round", 
              "comp", 
              "season", 
              "attendance", 
              "notes", 
              "captain", 
              "formation", 
              "referee", 
              "match report", 
              "notes"], 
             axis=1, inplace=True)

matches["team"].value_counts()

matches["date"] = pd.to_datetime(matches["date"])

matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["day_code"] = matches["date"].dt.dayofweek

#retrieve just hour from ko time - time of day may affect performance
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)

#create dict to rename teams so they match in home/away column
class MissingDict(dict):
    __missing__ = lambda self, key: key
    
map_values = {  
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Sheffield United": "Sheffield Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",    
}
mapping = MissingDict(**map_values)

matches['team'] = matches['team'].map(mapping)

matches.head()

#determine points acquired
matches["points"] = matches["result"].apply(lambda row: 3 if row=="W" else 1 if row=="D" else 0)

#convert W/L/D to numbers for classification
matches["results_class"] = matches["result"].apply(lambda row: 2 if row=="W" else 1 if row=="D" else 0)

matches.columns

matches.sort_values('date', inplace=True)

#create rolling averages based on previous 4 games
cols = ['points', 'gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'poss', 'xg', 'xga']
new_cols = [f"{c}_rolling" for c in cols]
matches[new_cols] = matches.groupby('team')[cols].transform(lambda x: x.rolling(4).mean().shift().bfill())

#group by team
grp_matches = matches.groupby("team").apply(lambda a: a[:]).drop('team', axis=1).droplevel(1)

grp_matches.reset_index(inplace=True)

#keep important columns that have potential impact on performance
avg_matches = grp_matches[['date', 
                           'team',
                           'opponent',
                           'venue_code', 
                           'hour', 
                           'day_code', 
                           'points_rolling', 
                           'gf_rolling', 
                           'ga_rolling', 
                           'sh_rolling', 
                           'sot_rolling', 
                           'dist_rolling', 
                           'fk_rolling', 
                           'pk_rolling', 
                           'pkatt_rolling',
                           'poss_rolling',
                           'xg_rolling', 
                           'xga_rolling', 
                           'results_class']].dropna(axis=0)

#split in to home and awya matches based on venue code
home_matches = avg_matches[avg_matches["venue_code"]==1].sort_values("date")
away_matches = avg_matches[avg_matches["venue_code"]==0].sort_values("date")

home_matches.head()

away_matches.head()

#remerge so no matches are repeated
merge_matches = pd.merge(home_matches, away_matches, 
                         left_on=["date", "team", "opponent"], 
                         right_on=["date", "opponent", "team"], 
                         suffixes=('_home', '_away')).sort_values("date")

merge_matches.drop(["opponent_home", "opponent_away", "venue_code_home", "venue_code_away", "results_class_away", "hour_away", "day_code_away"], axis=1, inplace=True)

merge_matches.columns

merge_matches.rename({"hour_home": "hour", "day_code_home": "day_code"}, axis=1, inplace=True)

merge_matches.columns

merge_matches["team_home_code"] = merge_matches["team_home"].astype("category").cat.codes

merge_matches["team_away_code"] = merge_matches["team_away"].astype("category").cat.codes

# create columns with average stat differences between the two teams
merge_matches['points_rolling_diff'] = (merge_matches['points_rolling_home']-merge_matches['points_rolling_away'])
merge_matches['gf_rolling_diff'] = (merge_matches['gf_rolling_home']-merge_matches['gf_rolling_away'])
merge_matches['ga_rolling_diff'] = (merge_matches['ga_rolling_home']-merge_matches['ga_rolling_away'])
merge_matches['sh_rolling_diff'] = (merge_matches['sh_rolling_home']-merge_matches['sh_rolling_away'])
merge_matches['sot_rolling_diff'] = (merge_matches['sot_rolling_home']-merge_matches['sot_rolling_away'])
merge_matches['poss_rolling_diff'] = (merge_matches['poss_rolling_home']-merge_matches['poss_rolling_away'])
merge_matches['xg_rolling_diff'] = (merge_matches['xg_rolling_home']-merge_matches['xg_rolling_away'])
merge_matches['xga_rolling_diff'] = (merge_matches['xga_rolling_home']-merge_matches['xga_rolling_away'])

#final column features
final_df = merge_matches[['date',
                          'hour', 
                          'day_code', 
                          'team_home',
                          'team_away', 
                          'points_rolling_diff', 
                          'gf_rolling_diff', 
                          'ga_rolling_diff', 
                          'sh_rolling_diff', 
                          'sot_rolling_diff', 
                          'poss_rolling_diff', 
                          'xg_rolling_diff', 
                          'xga_rolling_diff', 
                          'results_class_home']]


#observe correlations between features and target 
ML.corrs(final_df, "results_class_home")

#convert team names to numeric codes for machine learning model
final_df["team_home_code"] = final_df["team_home"].astype("category").cat.codes
final_df["team_away_code"] = final_df["team_away"].astype("category").cat.codes

#select only numeric features for model
final_df = final_df.select_dtypes(['number'])

final_df.head()



# =============================================================================
# ML Model Build
# =============================================================================

ML.split(final_df, "results_class_home")

ML.X_train
ML.y_train

from hyperopt import hp, STATUS_OK
from hyperopt.pyll import scope
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score, roc_auc_score, accuracy_score

ML.model = xgb.XGBClassifier()

space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
         'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 5, 1)),
         'n_estimators': scope.int(hp.quniform("n_estimators", 140, 160, 1)),
         'gamma': hp.quniform('gamma', 0.03, 0.07, 0.005),
         'learning_rate': hp.quniform('learning_rate', 0.4, 0.5, 0.01),
         'reg_lambda': hp.quniform('reg_lambda', 0.8, 0.85, 0.005),
         'early_stopping_rounds': hp.quniform('early_stopping_rounds', 80, 100, 1),
       }

def objective(space):

    model = xgb.XGBClassifier(objective="multi:softmax", 
                              num_class=3,
                              max_depth=space['max_depth'],
                              min_child_weight=space['min_child_weight'],
                              n_estimators=space['n_estimators'],
                              reg_lambda=space['reg_lambda'],
                              gamma=space['gamma'],
                              eval_metric="merror",
                              early_stopping_rounds=space['early_stopping_rounds']
                             )
    
    evaluation=[(ML.X_test, ML.y_test)]
    
    model.fit(ML.X_train, ML.y_train, eval_set=evaluation, verbose=False)
    
    # y_pred_probs = model.predict_proba(ML.X_test)
    y_pred = model.predict(ML.X_test)
    
    # score = accuracy_score(ML.y_test, y_pred)
    # score2 = roc_auc_score(ML.y_test, y_pred_probs, multi_class='ovo')
    # score = (0.7*score1+0.3*score2)/2
    score = ML.macro_f1(ML.y_test, y_pred)
    
    loss = 1-score
    
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

# space = {'max_depth': scope.int(hp.quniform("max_depth", 1, 10, 1)),
#        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
#        'n_estimators': scope.int(hp.quniform("n_estimators", 100, 500, 10)),
#        'early_stopping_rounds': hp.quniform('early_stopping_rounds', 1, 100, 2)
#        }

ML.optimise(objective, space, max_evals=1000)
ML.best_params

ML.getBestModelfromTrials()

ML.fit()

ML.predict("V4")
ML._predictions

ML.validate("V4")
ML._performance

ML.confusionMatrix()

import plotly.io as pio
pio.renderers.default='png'
ML.contour2D("max_depth", "min_child_weight", "n_estimators", 150)
ML.contour2D("max_depth", "early_stopping_rounds", "n_estimators", 150)
ML.contour2D("max_depth", "gamma", "n_estimators", 150)
ML.contour2D("max_depth", "reg_lambda", "n_estimators", 150)
ML.contour2D("max_depth", "learning_rate", "n_estimators", 150)


ML.featureImportance()

ML.learningCurve()


# =============================================================================
# Additional Analysis
# =============================================================================

df1 = pd.concat([ML.X_test, ML.y_test], axis=1).reset_index()
df2 = pd.concat([df1, pd.Series(ML._predictions["V4"], name="results_class_home_pred")], axis=1)
act_avgs = df2.groupby("team_home_code")["results_class_home"].mean()
pred_avgs = df2.groupby("team_home_code")["results_class_home_pred"].mean()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(act_avgs.index, act_avgs.values, color="darkblue", label="True")
plt.plot(pred_avgs.index, pred_avgs.values, color="darkturquoise", label="Prediction")
plt.legend()
plt.show()

import seaborn as sns
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(final_df.corr(), vmin=-1, vmax=1, annot=True, cmap='mako')

fig, ax = plt.subplots(1, figsize=(16,6))
bins = [0, 1, 2, 3]
n, bins, patches = plt.hist(final_df["results_class_home"], bins=[0,1,2,3], color="darkturquoise", edgecolor='black')
xticks=[0.5, 1.5, 2.5]
plt.xticks(xticks, ["L", "D", "W"])
plt.title('Histogram of Wins, Draws and Losses\n', loc = 'left', fontsize = 20)
plt.xlabel('\nResults Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
for idx, value in enumerate(n):
    if value > 0:
        plt.text(xticks[idx], value+5, int(value), ha='center')
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()      

features = ['points_rolling_diff', 'gf_rolling_diff', 'ga_rolling_diff', 'sh_rolling_diff', 'sot_rolling_diff', 
            'poss_rolling_diff', 'xg_rolling_diff', 'xga_rolling_diff']

fig, axs = plt.subplots(1, len(features), figsize=(20, 6))

# Loop through all features and plot the box plot for each feature
for i, feature in enumerate(features):
    # Plot the box plot for the feature
    axs[i].boxplot([final_df[final_df['results_class_home'] == 0][feature], final_df[final_df['results_class_home'] == 1][feature], final_df[final_df['results_class_home'] == 2][feature]], labels=['L', 'D', 'W'])

    # Add title and labels
    axs[i].set_title(f"''{feature}'")
    axs[i].set_xlabel("Class")
    axs[i].set_ylabel(feature)

# Show the plot
plt.show()

