import streamlit as st
import pandas as pd
import pickle

# Load the saved model and final_df
pipe = pickle.load(open('model_pipeline.pkl', 'rb'))
final_df = pickle.load(open('final_df.pkl', 'rb'))

# Streamlit app for IPL Win Predictor
st.title('IPL Win Predictor 2025')

# Current IPL teams (as of the latest season)
teams = sorted([
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
])

# Extract city info from final_df (keeping only unique cities)
cities = sorted(final_df['city'].unique())

# Create layout for team and city selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

selected_city = st.selectbox('Select the city', cities)

# Input for target and match situation
target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    wickets = st.number_input('Wickets', min_value=0, max_value=9)
with col5:
    overs = st.number_input('Overs completed', min_value=0, max_value=20)

# Predict the win probability when the button is pressed
if st.button('Predict Probability'):
    # Calculate additional match data
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets  # Adjusted to match model's expected column name
    crr = score / overs  # Current run rate
    rrr = (runs_left * 6) / balls_left  # Required run rate

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_remaining],  # Updated column name
        'total_run_x': [target],  # Updated column name
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make predictions using the loaded model
    result = pipe.predict_proba(input_df)

    # Extract and display win probabilities
    win_prob_batting_team = round(result[0][1] * 100)  # Probability for the batting team
    win_prob_bowling_team = round(result[0][0] * 100)  # Probability for the bowling team

    st.header('Winning Probability:')
    st.subheader(f"{batting_team} : {win_prob_batting_team} %")
    st.subheader(f"{bowling_team} : {win_prob_bowling_team} %")
