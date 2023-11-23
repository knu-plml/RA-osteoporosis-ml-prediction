params_lr = {
 'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]
}

params_rf = {
    'max_depth': range(10,101,30),
    'min_samples_split': range(1,11,3),
    'n_estimators': range(100,1001,300)
}

params_xgb = {
    'eta': [0.01, 0.05, 0.1, 0.15, 0.2],
    'min_child_weight': range(1,6,2), 
    'gamma':range(0,5),
    'subsample':[i/10.0 for i in range(5,10)],
    'colsample_bytree':[i/10.0 for i in range(5,10)],
    'max_depth': range(3,10,3),
    'eval_metric': ['error']
}

params_lgb = {
    'learning_rate': [i/10.0 for i in range(1,10)],
    'max_depth': range(3,10,3),
    'n_estimators': [100, 300, 500, 1000]
}
