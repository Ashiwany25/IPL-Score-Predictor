"""Streamlit interface for the IPL ANN score predictor."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'ipl_ann.keras'
SCALER = ROOT / 'models' / 'scaler.joblib'

st.set_page_config(page_title='IPL Score Predictor', page_icon='🏏', layout='centered')
st.title('🏏 IPL Score Predictor')
st.caption('ANN-powered estimate of the final first-innings score')
st.info('Educational prototype: the supplied training data is synthetic, not historical IPL data. Predictions are not validated for real matches.')

@st.cache_resource(show_spinner='Loading trained ANN…')
def load_assets():
    import joblib
    from tensorflow import keras
    return keras.models.load_model(MODEL), joblib.load(SCALER)

with st.form('score_form'):
    col1, col2 = st.columns(2)
    with col1:
        runs = st.number_input('Current runs', min_value=0, max_value=500, value=105, step=1)
        overs = st.number_input('Completed overs', min_value=0, max_value=19, value=12, step=1)
    with col2:
        wickets = st.number_input('Wickets lost', min_value=0, max_value=10, value=3, step=1)
        balls = st.number_input('Balls in current over', min_value=0, max_value=5, value=3, step=1)
    submitted = st.form_submit_button('Predict final score', type='primary', use_container_width=True)

if submitted:
    elapsed = int(overs) + int(balls) / 6
    if elapsed <= 0:
        st.error('Enter a match situation after at least one ball has been bowled.')
    elif not MODEL.is_file() or not SCALER.is_file():
        st.error('Trained model files are missing. Run `python generate_demo_data.py` then `python train.py` locally. For deployment, include the resulting models/ipl_ann.keras and models/scaler.joblib in your deployment source.')
    else:
        try:
            import numpy as np
            model, scaler = load_assets()
            features = np.array([[elapsed, runs, wickets, runs / elapsed]], dtype=np.float32)
            raw = float(model.predict(scaler.transform(features), verbose=0)[0][0])
            predicted = round(max(int(runs), raw))
            st.metric('Predicted final score', f'{predicted} runs')
            st.caption(f'Current: {runs}/{wickets} after {overs}.{balls} overs · Projected additional runs: {predicted - runs}')
            st.warning('This is a model estimate, not a guaranteed score. Synthetic-data predictions should not be treated as real IPL forecasts.')
        except Exception as exc:
            st.error(f'Unable to run prediction: {exc}')

with st.expander('How the prediction works'):
    st.write('The neural network uses elapsed overs, runs, wickets, and current run rate. Its input features are scaled with the same scaler used during training.')
    st.write('To improve the model, train and evaluate it on real, properly sourced historical IPL match data.')
