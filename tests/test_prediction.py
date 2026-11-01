import numpy as np
import pytest
from config import FEATURES
from src.prediction import predict_strength,validate_inputs


def sample(bundle):
    return bundle['data'].iloc[bundle['train_indices'][0]][FEATURES].to_dict()


def test_prediction_consistent_with_pipeline(bundle):
    values=sample(bundle)
    result=predict_strength(bundle,values)
    expected=bundle['models'][bundle['metadata']['selected']].predict(validate_inputs(values))[0]
    assert result['Prediction (MPa)']==expected
    assert result['Water/Cement']==values['Water']/values['Cement']
    assert result['warnings']==[]


@pytest.mark.parametrize('feature,value',[('Cement',0),('Water',0),('Age',0),('Age',28.5),
    ('Cement',-10),('Fine Aggregate',np.inf),('Fly Ash',np.nan),('Water','200'),('Water',True),('Water',None)])
def test_invalid_inputs(bundle,feature,value):
    values=sample(bundle);values[feature]=value
    with pytest.raises(ValueError): predict_strength(bundle,values)


def test_schema_and_extrapolation(bundle):
    values=sample(bundle)
    with pytest.raises(ValueError):validate_inputs({'Cement':100})
    with pytest.raises(ValueError):validate_inputs({**values,'extra':0})
    values['Age']=1000
    result=predict_strength(bundle,values)
    assert any('Outside training range' in w for w in result['warnings'])
