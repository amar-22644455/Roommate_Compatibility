# backend/ml/matching.py
import pandas as pd
import networkx as nx
import itertools
from backend.ml.feature_prep import create_pairwise_features


def find_optimal_roommates(df_students, model):
    """
    Takes the uploaded students dataframe and the loaded RandomForest model,
    and returns a structured list of optimal roommate pairs.
    """
    print("Generating all possible student pairs...")
    student_ids = df_students['student_id'].tolist()

    # Generate all unique pairs
    all_pairs = list(itertools.combinations(student_ids, 2))
    df_all_pairs = pd.DataFrame(all_pairs, columns=['student_id_A', 'student_id_B'])

    # Engineer features for prediction
    X_predict = create_pairwise_features(df_students, df_all_pairs)

    print("Predicting compatibility scores for all pairs...")
    
    # Drop IDs before feeding into the model (ensure this matches training!)
    features_only = X_predict.drop(columns=['student_id_A', 'student_id_B'])
    scores = model.predict(features_only)
    df_all_pairs['predicted_score'] = scores

    print("Computing optimal roommate pairs using Maximum Weight Matching...")

    G = nx.Graph()
    for _, row in df_all_pairs.iterrows():
        G.add_edge(int(row['student_id_A']), int(row['student_id_B']), weight=row['predicted_score'])

    matching = nx.max_weight_matching(G, maxcardinality=True)

    # ---------------------------------------------------------
    # NEW: Format the output for the FastAPI JSON response
    # ---------------------------------------------------------
    assignments = []
    total_system_score = 0

    for student_a, student_b in matching:
        score = G[student_a][student_b]['weight']
        total_system_score += score
        
        # Append as a dictionary for easy JSON serialization
        assignments.append({
            "student_1": student_a,
            "student_2": student_b,
            "compatibility_score": round(score, 2)
        })

    avg_score = total_system_score / len(matching) if matching else 0
    
    return {
        "average_hostel_score": round(avg_score, 2),
        "total_pairs": len(assignments),
        "assignments": assignments
    }