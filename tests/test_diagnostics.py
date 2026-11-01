import numpy as np
import statsmodels.api as sm
from config import FEATURES
from src.diagnostics import calculate_diagnostics
from src.feature_analysis import associations,observed_trend
from src.feature_engineering import with_analysis_features


def test_vif_intercept_and_residual_definition(bundle):
    results=calculate_diagnostics(bundle)
    X=bundle['data'].iloc[bundle['train_indices']][FEATURES]
    first=FEATURES[0]
    aux=sm.OLS(X[first],sm.add_constant(X.drop(columns=[first]))).fit()
    assert np.isclose(results['vif'].iloc[0].VIF,1/(1-aux.rsquared))
    pred=bundle['predictions'][bundle['metadata']['selected']]
    np.testing.assert_allclose(pred.Residual,pred.Actual-pred.Predicted)
    assert 0<=results['statistics']['BP p']<=1
    assert 0<=results['statistics']['Shapiro p']<=1


def test_observed_curves_finite(bundle):
    data=with_analysis_features(bundle['data'])
    stats=associations(data)
    assert stats.n.eq(len(data)).all()
    for feature in ['Age','Water_Cement_Ratio']:
        curve=observed_trend(data,feature).data[-1]
        assert np.isfinite(curve.y).all()
        assert np.isfinite(curve.x).all()
