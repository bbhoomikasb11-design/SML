import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from config import FEATURES,TARGET
from src.evaluation import regression_metrics
from src.model_training import run_experiment
from src.preprocessing import make_pipeline


def test_no_group_overlap_and_validation_coverage(bundle):
    train,test=bundle['train_indices'],bundle['test_indices']
    groups=bundle['groups']
    assert not set(groups[train]) & set(groups[test])
    assert sorted(np.r_[train,test].tolist())==list(range(len(bundle['data'])))
    validation=[]
    for fit,val in bundle['cv_splits']:
        assert not set(groups[train][fit]) & set(groups[train][val])
        assert sorted(np.r_[fit,val].tolist())==list(range(len(train)))
        validation.extend(val.tolist())
    assert sorted(validation)==list(range(len(train)))


def test_preprocessing_fitted_only_on_training_and_cv_scores(bundle):
    X=bundle['data'].iloc[bundle['train_indices']][FEATURES]
    y=bundle['data'].iloc[bundle['train_indices']][TARGET]
    for name,pipeline in bundle['models'].items():
        np.testing.assert_allclose(pipeline.named_steps['impute'].statistics_,X.median())
        np.testing.assert_allclose(pipeline.named_steps['scale_inputs'].mean_,X.mean())
        measured=[]
        for fit,val in bundle['cv_splits']:
            local=clone(pipeline).fit(X.iloc[fit],y.iloc[fit])
            np.testing.assert_allclose(local.named_steps['impute'].statistics_,X.iloc[fit].median())
            measured.append(regression_metrics(y.iloc[val],local.predict(X.iloc[val]))['RMSE'])
        saved=bundle['cv_scores'].query('Model == @name').RMSE
        np.testing.assert_allclose(measured,saved,rtol=1e-8)


def test_all_metrics_and_saved_model_round_trip(bundle,tmp_path):
    Xtest=bundle['data'].iloc[bundle['test_indices']][FEATURES]
    ytest=bundle['data'].iloc[bundle['test_indices']][TARGET]
    for name,pipeline in bundle['models'].items():
        pred=pipeline.predict(Xtest)
        assert np.isfinite(pred).all()
        measured=regression_metrics(ytest,pred)
        row=bundle['metrics'].set_index('Model').loc[name]
        for metric,value in measured.items(): assert np.isclose(value,row[f'Test {metric}'])
        path=tmp_path/f'{name}.joblib';joblib.dump(pipeline,path)
        np.testing.assert_allclose(pred,joblib.load(path).predict(Xtest))
    expected=bundle['metrics'].sort_values(['CV RMSE','Terms']).iloc[0].Model
    assert bundle['metadata']['selected']==expected


def test_reproducible_experiment(bundle):
    rerun=run_experiment(save=False)
    pd.testing.assert_frame_equal(rerun['metrics'],bundle['metrics'],rtol=1e-8)
    np.testing.assert_array_equal(rerun['train_indices'],bundle['train_indices'])


def test_missing_value_imputation_without_validation_leakage():
    X=pd.DataFrame([[1.,2.],[3.,np.nan],[5.,6.]],columns=['a','b'])
    for degree in [1,2,3]:
        pipe=make_pipeline(degree).fit(X,[2.,4.,8.])
        stats=pipe.named_steps['impute'].statistics_.copy()
        assert np.isfinite(pipe.predict(pd.DataFrame([[999.,np.nan]],columns=X.columns))).all()
        np.testing.assert_array_equal(stats,[3.,4.])
