# backend/ml/feature_prep.py
import pandas as pd
import numpy as np

categorical_cols = [
    'gender', 'department', 'study_noise_preference', 'fan_or_cooler_preference',
    'study_habit', 'food_preference', 'exam_preparation_style', 'social_frequency',
    'relationship_status', 'career_interest', 'cult_sports', 'language'
]

numerical_cols = [
    'year_of_study', 'sleep_time', 'wake_up_time', 'alarm_usage', 
    'morning_productivity', 'night_productivity', 'cleanliness_score', 
    'room_organization_level', 'noise_tolerance', 'daily_study_hours', 
    'introvert_extrovert_score', 'smoking_drinking', 'workout', 
    'gaming', 'anime', 'room_stay_duration'
]

def create_pairwise_features(df_students, df_pairs):
    """Merges student profiles and computes absolute differences & similarities."""
    
    # Note: Make sure the suffixes here (_1 and _2) match what you use 
    # in your calculate_compatibility function!
    df = df_pairs.merge(df_students, left_on='student_id_A', right_on='student_id', suffixes=('', '_1'))
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
        
    df = df.merge(df_students, left_on='student_id_B', right_on='student_id', suffixes=('_1', '_2'))
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])

    features = pd.DataFrame()
    features['student_id_A'] = df['student_id_A']
    features['student_id_B'] = df['student_id_B']

    # Numerical differences (Absolute difference)
    for col in numerical_cols:
        features[f'diff_{col}'] = np.abs(df[f'{col}_1'] - df[f'{col}_2'])

    # Categorical similarities (1 if same, 0 if different)
    for col in categorical_cols:
        features[f'sim_{col}'] = (df[f'{col}_1'] == df[f'{col}_2']).astype(int)

    # Add target variable if it exists (for training phase)
    if 'compatibility_score' in df.columns:
        features['compatibility_score'] = df['compatibility_score']

    return features