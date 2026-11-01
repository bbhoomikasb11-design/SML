import numpy as np
import pandas as pd
import pytest
from config import FEATURES, TARGET
from src.data_loader import audit_data, load_data, prepare_data
from src.feature_engineering import with_analysis_features


def test_original_data_contract():
    data=load_data()
    audit=audit_data(data)
    assert data.shape==(1030,9)
    assert list(data)==FEATURES+[TARGET]
    assert audit['duplicate_rows']==25
    assert audit['invalid_rows']==0
    assert sum(audit['missing'].values())==0
    assert len(prepare_data(data))==1005
    assert data.iloc[0].Cement==540


@pytest.mark.parametrize('column,value',[('Water',-1),('Cement',0),('Age',2.5),('Strength',0),('Fly Ash',np.inf)])
def test_invalid_dataset_rejected(column,value):
    data=load_data().head(4).copy()
    data.loc[data.index[0],column]=value
    assert audit_data(data)['invalid_rows']==1
    with pytest.raises(ValueError,match='Invalid'): prepare_data(data)


def test_missing_predictors_retained_targets_dropped():
    data=load_data().head(4).copy()
    data.loc[0,'Water']=np.nan
    data.loc[1,TARGET]=np.nan
    prepared=prepare_data(data)
    assert 1 not in prepared.index
    assert pd.isna(prepared.loc[0,'Water'])


def test_feature_ratio_and_age_bins(bundle):
    enriched=with_analysis_features(bundle['data'])
    np.testing.assert_allclose(enriched.Water_Cement_Ratio,enriched.Water/enriched.Cement)
    assert enriched['Age Group'].notna().all()
    assert 'Water_Cement_Ratio' not in bundle['data']
