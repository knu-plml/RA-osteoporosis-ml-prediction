import pandas as pd
import numpy as np
import scipy.stats as st
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# Scoring
def score_print(acc, auc, f1, txt):
    print('*' * 50)
    print(txt)
    print()
    
    print('ACC by CV: ', np.round(acc, 4))
    print('Average validation ACC: ', np.round(np.mean(acc), 4))
    print()
   
    print('AUROC by CV: ', np.round(auc, 4))
    print('Average validation AUROC: ', np.round(np.mean(auc), 4))
    print('95% CI: ', st.t.interval(0.95, df=len(auc)-1, loc=np.mean(auc), scale=st.sem(auc)))
    print()
    
    print('F1-score by CV: ', np.round(f1, 4))
    print('Average validation F1-score: ', np.round(np.mean(f1), 4))
    
    print('*' * 50)
    print()
    

# Plot SHAP
def shap_val(model, X, X_train, X_test, y_train, txt):
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    fig = plt.gcf()
    if len(np.array(shap_values).shape) != 2:
        k = shap_values[1]
    else:
        k = shap_values
    shap.summary_plot(k, X_test, show=False)
    
    plt.savefig('./shap_' + txt + '.png', dpi=300)
    plt.clf()
    

# Plot ROC curve
def roc_curve_plot(lr, rf, xgb, lgb, X, y):
    models = [lr, rf, xgb, lgb]
    header = ['LR', 'RF', 'XGB', 'LGB']
    clr = ['blue', 'green', 'red', 'yellow']
    
    cv = KFold(n_splits=5, shuffle = True, random_state=0)
    
    plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'black')
    
    for m, h, c in zip(models, header, clr):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0,1,100)

        for train, test in cv.split(X, y):
            prediction = m.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
            fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color = c,
                 label = h + ' (AUC = %0.3f)' % (mean_auc),
                 lw = 1, alpha = 1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.rc('font', size=15)
    plt.rc('axes', labelsize=15)   
    plt.rc('xtick', labelsize=13)  
    plt.rc('ytick', labelsize=13) 
    plt.rc('legend', fontsize=13)  
    plt.rc('figure', titlesize=13) 

    plt.savefig('./AUROC.png', dpi=300)
    

# Plot ICC score
def plot_ICC(icc_result):
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Type', y='ICC', data=icc_result)
    plt.title('Intraclass Correlation Coefficients')
    plt.ylim(0, 1)
    plt.ylabel('ICC Value')
    plt.xlabel('ICC Type')
    plt.savefig('./ICC.png', dpi=300)
    
    
# Plot ICC figure
def plot_ICC_fig(dfp_lr, dfp_rf, dfp_xgb, dfp_lgb):
    x = y = np.arange(5)
    plt.figure(figsize=(20,8))

    plt.scatter(range(len(dfp_lr['targets'].iloc[:30])), dfp_lr['ratings'].iloc[:30], s=100, marker='o', color='r', linewidths=1, label='LR')
    plt.scatter(range(len(dfp_rf['targets'].iloc[:30])), dfp_rf['ratings'].iloc[:30], s=100, marker='o', color='g', linewidths=1, label='RF')
    plt.scatter(range(len(dfp_xgb['targets'].iloc[:30])), dfp_xgb['ratings'].iloc[:30], s=100, marker='o', color='c', linewidths=1, label='XGB')
    plt.scatter(range(len(dfp_lgb['targets'].iloc[:30])), dfp_lgb['ratings'].iloc[:30], s=100, marker='o', color='b', linewidths=1, label='LGBM')
    
    plt.legend(fontsize=18)
    plt.grid(True, axis='x')
    plt.xticks(np.arange(0, 30, 1), fontsize=18)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=18)
    plt.xlabel('Patient ID', fontsize=23, labelpad = 12)
    plt.ylabel('Prediction probability', fontsize=23, labelpad = 12)

    plt.savefig('./ICC_fig.png', dpi=300)