# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os

# Import your ML pipeline modules
from backend.ml.matching import find_optimal_roommates
from backend.ml.model_loader import get_model

app = FastAPI(title="Hostel Roommate Matching System")

# 1. Setup CORS so React can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths for storing data locally
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# The static paths for your input and output files
STUDENTS_CSV_FILE = os.path.join(DATA_DIR, "students.csv")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")


@app.get("/")
def read_root():
    return {"message": "Roommate Matching API is running."}


# 2. ADMIN ENDPOINT: Trigger the matching process from the local CSV
@app.post("/api/admin/run-matching")
async def run_matching():
    # Verify you actually placed the file in the right spot before running
    if not os.path.exists(STUDENTS_CSV_FILE):
        raise HTTPException(
            status_code=404, 
            detail=f"Data file not found. Please place 'students.csv' inside {DATA_DIR}."
        )
    
    try:
        print(f"Reading student data from {STUDENTS_CSV_FILE}...")
        df_students = pd.read_csv(STUDENTS_CSV_FILE)
        
        # Load the trained model using your Singleton loader
        model = get_model()
        
        # Run the matching algorithm
        print("Starting the NetworkX matching pipeline...")
        matching_results = find_optimal_roommates(df_students, model)
        
        # Save the results to a JSON file for fast student lookups
        with open(ASSIGNMENTS_FILE, "w") as f:
            json.dump(matching_results, f, indent=4)
            
        return {
            "message": "Roommate matching completed successfully!",
            "summary": {
                "total_pairs": matching_results["total_pairs"],
                "average_hostel_score": matching_results["average_hostel_score"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")


# 3. STUDENT ENDPOINT: Search for assigned roommate
@app.get("/api/student/match/{student_id}")
async def get_roommate(student_id: int):
    # Check if the admin has run the matching yet
    if not os.path.exists(ASSIGNMENTS_FILE):
        raise HTTPException(
            status_code=404, 
            detail="Roommate assignments have not been generated yet. Admin must run the matching pipeline first."
        )
        
    with open(ASSIGNMENTS_FILE, "r") as f:
        data = json.load(f)
        
    assignments = data.get("assignments", [])
    
    # Search for the student in the pairs
    for pair in assignments:
        if pair["student_1"] == student_id:
            return {"your_id": student_id, "roommate_id": pair["student_2"], "compatibility_score": pair["compatibility_score"]}
        elif pair["student_2"] == student_id:
            return {"your_id": student_id, "roommate_id": pair["student_1"], "compatibility_score": pair["compatibility_score"]}
            
    # If the loop finishes without returning, the student ID wasn't found
    raise HTTPException(status_code=404, detail=f"No roommate assignment found for student ID {student_id}.")