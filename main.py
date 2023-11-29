from utils import *
from params import *

# Data load - Global
df = pd.read_csv('./DATA.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


def main():
    # Model init
    lr, rf, xgb, lgb = model_init()

    # Hyperparameter search
    best_params_lr, best_params_rf, best_params_xgb, best_params_lgb = HP_search(lr, rf, xgb, lgb)

    # Best model
    lr, rf, xgb, lgb = best_model(best_params_lr, best_params_rf, best_params_xgb, best_params_lgb)

    # Cross Validation
    acc_lr, roc_lr, f1_lr = CV(lr)
    acc_rf, roc_rf, f1_rf = CV(rf)
    acc_xgb, roc_xgb, f1_xgb = CV(xgb)
    acc_lgb, roc_lgb, f1_lgb = CV(lgb)

    # Print scores
    score_print(acc_lr, roc_lr, f1_lr, 'Logistic Regression')
    score_print(acc_rf, roc_rf, f1_rf, 'Random Forest')
    score_print(acc_xgb, roc_xgb, f1_xgb, 'XGBoost')
    score_print(acc_lgb, roc_lgb, f1_lgb, 'LightGBM')

    # SHAP
    lr, rf, xgb, lgb = best_model(best_params_lr, best_params_rf, best_params_xgb, best_params_lgb)
    shap_val(rf, X, X_train, X_test, y_train, 'rf')
    shap_val(xgb, X, X_train, X_test, y_train, 'xgb')
    shap_val(lgb, X, X_train, X_test, y_train, 'lgb')

    # AUROC
    lr, rf, xgb, lgb = best_model(best_params_lr, best_params_rf, best_params_xgb, best_params_lgb)
    roc_curve_plot(lr, rf, xgb, lgb, X, y)

    # ICC
    lr, rf, xgb, lgb = best_model(best_params_lr, best_params_rf, best_params_xgb, best_params_lgb)
    Calc_ICC(lr, rf, xgb, lgb)


# Models - init
def model_init():
    lr = LogisticRegression()
    rf = RandomForestClassifier()
    xgb = XGBClassifier(eval_metric='mlogloss')
    lgb = LGBMClassifier()

    return lr, rf, xgb, lgb


# Hyperparameter search
def HP_search(lr, rf, xgb, lgb):
    # Hyperparameter search - Logistic Regression
    grid_search = GridSearchCV(lr, param_grid=params_lr, cv=2, scoring='accuracy')
    grid_search.fit(X, y)
    best_params_lr = grid_search.best_params_

    # Hyperparameter search - Random Forest
    random_search = RandomizedSearchCV(rf, param_distributions=params_rf, cv=5, scoring='accuracy')
    random_search.fit(X, y)
    best_params_rf = random_search.best_estimator_

    # Hyperparameter search - XGBoost
    random_search = RandomizedSearchCV(xgb, param_distributions=params_xgb, cv=5, scoring='accuracy')
    random_search.fit(X, y)
    best_params_xgb = random_search.best_estimator_

    # Hyperparameter search - LightGBM
    random_search = RandomizedSearchCV(lgb, param_distributions=params_lgb, cv=5, scoring='accuracy')
    random_search.fit(X, y)
    best_params_lgb = random_search.best_estimator_

    return best_params_lr, best_params_rf, best_params_xgb, best_params_lgb


# Best models
def best_model(best_params_lr, best_params_rf, best_params_xgb, best_params_lgb):
    lr = LogisticRegression(C=best_params_lr['C'])
    rf = RandomForestClassifier(
         max_depth=best_params_rf.max_depth,
         min_samples_split=best_params_rf.min_samples_split,
         n_estimators=best_params_rf.n_estimators)
    xgb = XGBClassifier(
          learning_rate=best_params_xgb.learning_rate,
          min_child_weight=best_params_xgb.min_child_weight,
          gamma=best_params_xgb.gamma,
          subsample=best_params_xgb.subsample,
          colsample_bytree=best_params_xgb.colsample_bytree,
          max_depth=best_params_xgb.max_depth,
          eval_metric='error')
    lgb = LGBMClassifier(learning_rate=best_params_lgb.learning_rate,
                         max_depth=best_params_lgb.max_depth,
                         n_estimators=best_params_lgb.n_estimators)

    return lr, rf, xgb, lgb


# Cross validation
def CV(model):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    acc = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
    roc = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold)
    f1 = cross_val_score(model, X, y, scoring='f1', cv=kfold)

    return acc, roc, f1


# Calc ICC
def Calc_ICC(lr, rf, xgb, lgb):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    df_fu = pd.DataFrame(columns=['targets', 'raters', 'ratings', 'labels'])

    for idx, (train, test) in enumerate(cv.split(X, y)):
        df_in = pd.DataFrame()
        pred_lr = lr.fit(X.iloc[train].drop('PID', axis=1), Y.iloc[train].values).predict_proba(X.iloc[test].drop('PID', axis=1))
        pred_rf = rf.fit(X.iloc[train].drop('PID', axis=1), Y.iloc[train].values).predict_proba(X.iloc[test].drop('PID', axis=1))
        pred_xgb = xgb.fit(X.iloc[train].drop('PID', axis=1), Y.iloc[train].values).predict_proba(X.iloc[test].drop('PID', axis=1))
        pred_lgb = lgb.fit(X.iloc[train].drop('PID', axis=1), Y.iloc[train].values).predict_proba(X.iloc[test].drop('PID', axis=1))

        ratings = np.concatenate([pred_lr[:, 1], pred_rf[:, 1], pred_xgb[:, 1], pred_lgb[:, 1]])
        raters = ['lr'] * len(pred_lr) + ['rf'] * len(pred_rf) + ['xgb'] * len(pred_xgb) + ['lgb'] * len(pred_lgb)

        idd = X.iloc[test]['PID'].index.values
        targets = np.concatenate([idd, idd, idd, idd])
        labels = y.iloc[test].values
        labels = np.concatenate([labels, labels, labels, labels])

        df_in['targets'] = targets
        df_in['raters'] = raters
        df_in['ratings'] = ratings
        df_in['labels'] = labels
        df_fu = pd.concat([df_fu, df_in])

    df_fu['targets'] = df_fu['targets'].astype(int)
    df_fu['ratings'] = df_fu['ratings'].astype(float)

    icc_result = pg.intraclass_corr(data=df_fu, targets='targets', raters='raters', ratings='ratings')
    plot_ICC(icc_result)

    df_proba = df_fu
    dfp_lr = df_proba[df_proba['raters'] == 'lr']
    dfp_rf = df_proba[df_proba['raters'] == 'rf']
    dfp_xgb = df_proba[df_proba['raters'] == 'xgb']
    dfp_lgb = df_proba[df_proba['raters'] == 'lgb']

    plot_ICC_fig(dfp_lr, dfp_rf, dfp_xgb, dfp_lgb)


if __name__ == "__main__":
    main()
