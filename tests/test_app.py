import pytest
from streamlit.testing.v1 import AppTest
from components.sidebar import PAGES


@pytest.fixture
def app():
    return AppTest.from_file('app.py',default_timeout=60).run()


def test_all_navigation_sections(app):
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
        assert len(app.title)==1


def button(app,label):
    return next(b for b in app.button if b.label==label)


def test_prediction_history_presets_reset_invalid(app):
    app.sidebar.radio[0].set_value('Strength Prediction').run()
    button(app,'Predict compressive strength').click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['history'])==1
    app.number_input(key='input_Cement').set_value(0)
    button(app,'Predict compressive strength').click().run()
    assert app.error and len(app.session_state['history'])==1
    button(app,'Reset inputs').click().run()
    assert app.number_input(key='input_Cement').value>0
    app.selectbox[0].set_value('7-day mixture')
    button(app,'Load preset').click().run()
    assert app.number_input(key='input_Age').value==7
    button(app,'Clear session history').click().run()
    assert len(app.session_state['history'])==0


def test_session_training_and_report_generation(app):
    app.sidebar.radio[0].set_value('Model Training').run()
    app.selectbox[0].set_value(3)
    button(app,'Train & compare all models').click().run()
    assert not app.exception and not app.error
    assert app.session_state['experiment']['metadata']['folds']==3
    app.sidebar.radio[0].set_value('Project Report').run()
    button(app,'Generate PDF & editable Word report').click().run(timeout=90)
    assert not app.exception and not app.error
    assert len(app.get('download_button'))==2


def test_empty_dataset_filter(app):
    app.sidebar.radio[0].set_value('Dataset Explorer').run()
    app.slider[0].set_range(365,365)
    app.slider[1].set_range(80.,82.6).run()
    assert not app.exception
    assert app.warning
