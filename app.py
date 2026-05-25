from flask import Flask, render_template, request, redirect, jsonify
import json
import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import random

from dotenv import load_dotenv
import os

load_dotenv()




# --- 1. LOCAL ML WORKOUT PLANNER (Random Forest) ---
X_train_workout = np.array([
    [0, 0, 2], [0, 1, 2], [0, 2, 2], [0, 3, 2], 
    [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], 
    [2, 0, 1], [2, 1, 1], [2, 2, 1], [2, 3, 1], 
    [0, 2, 1], [1, 2, 1], [1, 0, 2], [0, 0, 0]  
])
y_train_workout = np.array([0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 2, 2, 0])

ml_workout_planner = RandomForestClassifier(n_estimators=15, random_state=42)
ml_workout_planner.fit(X_train_workout, y_train_workout)


# --- 2. LOCAL ML DIET RECOMMENDER (KNN) ---
# When you have your dataset, delete the dummy data below and uncomment this line:
# diet_df = pd.read_csv('my_meal_dataset.csv')

# --- 2. LOCAL ML DIET RECOMMENDER (KNN) ---
# Load the real dataset from your CSV filea
# --- 2. LOCAL STRICT DIET RECOMMENDER ---
# Load the real dataset from your CSV file
try:
    diet_df = pd.read_csv('my_meal_dataset.csv')
except FileNotFoundError:
    print("⚠️ Dataset not found! Make sure my_meal_dataset.csv is in the same folder.")
    diet_df = pd.DataFrame(columns=['Meal_Name', 'Goal_Encoded', 'Diet_Encoded', 'Slot_Encoded'])

# --- WORKOUT DATASET ---
# Load workout dataset
try:
    workout_df = pd.read_csv("workout_dataset.csv")
except FileNotFoundError:
    print("⚠️ workout_dataset.csv not found")
    workout_df = pd.DataFrame(columns=["Exercise","Muscle","Type","Goal_Encoded","Sets","Reps"])

def generate_ml_weekly_diet(goal_encoded, diet_encoded):
    import random
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    slots_names = ["Breakfast", "Brunch", "Lunch", "Snacks", "Dinner"]
    
    weekly_plan = {day: [] for day in days}
    
    # 🚨 THE "DOCTOR" FILTER: Exact matches only. No borrowing from other diets!
    strict_meals = diet_df[(diet_df['Goal_Encoded'] == goal_encoded) & (diet_df['Diet_Encoded'] == diet_encoded)]
    
    for slot_encoded, slot_name in enumerate(slots_names):
        # Find all meals that perfectly match the Goal, Diet, and Time of Day
        slot_meals = strict_meals[strict_meals['Slot_Encoded'] == slot_encoded]['Meal_Name'].tolist()
        
        if not slot_meals:
            slot_meals = [f"Healthy {slot_name}"] # Safety fallback
            
        # Multiply the meals so we have enough to fill a 7-day week
        while len(slot_meals) < 7:
            slot_meals.extend(slot_meals)
            
        # Shuffle them so you don't eat the exact same thing on consecutive days
        random.shuffle(slot_meals)
        selected_meals = slot_meals[:7]
        
        for i, day in enumerate(days):
            weekly_plan[day].append((slot_name, selected_meals[i]))

    # Build the HTML Tracker (Checkboxes Removed)
    html = "<h4 style='margin-top:25px; color:#334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;'>📅 Weekly Diet Plan</h4>"
    for day in days:
        html += f"<div style='margin-bottom: 15px; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; background: #f8fafc;'>"
        html += f"<h5 style='margin-top: 0; margin-bottom: 12px; color: #6366f1; font-size: 1.05rem;'>{day}</h5>"
        
        html += "<ul style='list-style-type: none; padding-left: 0; margin: 0;'>"
        for slot_name, meal_name in weekly_plan[day]:
            html += f"<li style='margin-bottom: 8px; font-size: 0.95rem; color: #475569;'>"
            html += f"<strong style='color: #0f172a;'>{slot_name}:</strong> <span style='font-weight: 400;'>{meal_name}</span>"
            html += f"</li>"
        html += "</ul>"
        
        html += "</div>"
    
    return html

# --- WORKOUT GENERATOR ---
def generate_ml_weekly_workout(workout_plan_id, goal_encoded):

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    splits = {
        0:["Full Body","Cardio","Full Body","Cardio","Full Body","Cardio","Rest"],
        1:["Chest","Back","Legs","Shoulders","Arms","Cardio","Rest"],
        2:["Upper","Lower","Rest","Upper","Lower","Cardio","Rest"]
    }

    split = splits.get(workout_plan_id, splits[1])

    html = "<h3 style='margin-top:25px'>🏋️ Weekly Workout Plan</h3>"

    goal_exercises = workout_df[workout_df["Goal_Encoded"]==goal_encoded]
    if len(goal_exercises) == 0:
     return "<p>No workout data available for this goal.</p>"

    for i,day in enumerate(days):

        muscle = split[i]

        html += f"<div style='margin-bottom:15px;border:1px solid #e2e8f0;padding:15px;border-radius:8px;background:#f8fafc;'>"
        html += f"<h4 style='color:#6366f1'>{day} — {muscle}</h4>"

        if muscle == "Rest":
            html += "<p>Rest Day</p></div>"
            continue

        if muscle == "Full Body":
            selected = goal_exercises.sample(min(5,len(goal_exercises)))
        else:
             muscle_exercises = goal_exercises[goal_exercises["Muscle"] == muscle]

             if len(muscle_exercises) >= 4:
                selected = muscle_exercises.sample(4)

             elif len(muscle_exercises) > 0:
                selected = muscle_exercises

             else:
                selected = goal_exercises.sample(min(4, len(goal_exercises)))

        html += "<ul style='list-style:none;padding-left:0;'>"

        for _,row in selected.iterrows():

            html += f"""
            <li>
            <strong>{row['Exercise']}</strong><br>
            Sets: {row['Sets']}<br>
            Reps: {row['Reps']}
            </li>
            """

        html += "</ul></div>"

    return html


# --- COMBINED WORKOUT + DIET PLAN ---
def get_ml_plan_template(workout_plan_id, goal_encoded, diet_encoded):

    workout_html = generate_ml_weekly_workout(workout_plan_id, goal_encoded)

    diet_html = generate_ml_weekly_diet(goal_encoded, diet_encoded)

    return workout_html + diet_html

app = Flask(__name__)

USERS_FILE = "users.json"
WORKOUT_FILE = "workouts.json"

# --- 1. AI INITIALIZATION (Your Custom Code) ---
try:
    api_key = os.getenv("gemini_api")  # Hardcoded key provided

    if not api_key:
        print("!!! WARNING: API KEY IS NOT SET.")
        ai_model = None
    else:
        genai.configure(api_key=api_key)
        model_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # NOTE: If 'gemma-3-4b-it' fails, try 'gemini-1.5-flash' which is standard
        try:
            ai_model = genai.GenerativeModel(
                model_name="models/gemma-3-4b-it", 
                generation_config=model_config,
                safety_settings=safety_settings
            )
            print("✅ Gemma AI Model initialized.")
        except Exception as inner_e:
            print(f"⚠️ Gemma model failed ({inner_e}), falling back to Gemini Flash...")
            ai_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini Flash Model initialized as fallback.")

except Exception as e:
    print(f"!!! ERROR initializing AI Model: {e}")
    ai_model = None


# ----------------- Helpers -----------------
def load_data(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_data(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

EXERCISES = {
    # --- CHEST ---
    "Bench Press":            {"movement": "horizontal_push", "met": 6.0, "muscle": "Chest", "type": "compound"},
    "Incline Bench Press":    {"movement": "horizontal_push", "met": 6.0, "muscle": "Chest", "type": "compound"},
    "Decline Bench Press":    {"movement": "horizontal_push", "met": 6.0, "muscle": "Chest", "type": "compound"},
    "Dumbbell Press":         {"movement": "horizontal_push", "met": 5.5, "muscle": "Chest", "type": "compound"},
    "Incline Dumbbell Press": {"movement": "horizontal_push", "met": 5.5, "muscle": "Chest", "type": "compound"},
    "Chest Fly (Dumbbell)":   {"movement": "isolation",       "met": 4.0, "muscle": "Chest", "type": "isolation"},
    "Cable Crossover":        {"movement": "isolation",       "met": 4.0, "muscle": "Chest", "type": "isolation"},
    "Pec Deck / Machine Fly": {"movement": "isolation",       "met": 3.5, "muscle": "Chest", "type": "isolation"},
    "Push Ups":               {"movement": "horizontal_push", "met": 6.0, "muscle": "Chest", "type": "compound"},
    "Dips (Chest Focus)":     {"movement": "vertical_push",   "met": 6.5, "muscle": "Chest", "type": "compound"},

    # --- BACK ---
    "Pull Ups":               {"movement": "vertical_pull",   "met": 8.0, "muscle": "Back", "type": "compound"},
    "Chin Ups":               {"movement": "vertical_pull",   "met": 8.0, "muscle": "Back", "type": "compound"},
    "Lat Pulldown":           {"movement": "vertical_pull",   "met": 6.0, "muscle": "Back", "type": "compound"},
    "Barbell Row":            {"movement": "horizontal_pull", "met": 6.5, "muscle": "Back", "type": "compound"},
    "Dumbbell Row":           {"movement": "horizontal_pull", "met": 6.0, "muscle": "Back", "type": "compound"},
    "Seated Cable Row":       {"movement": "horizontal_pull", "met": 5.5, "muscle": "Back", "type": "compound"},
    "T-Bar Row":              {"movement": "horizontal_pull", "met": 6.5, "muscle": "Back", "type": "compound"},
    "Face Pulls":             {"movement": "horizontal_pull", "met": 4.5, "muscle": "Back", "type": "isolation"},
    "Shrugs":                 {"movement": "isolation",       "met": 4.0, "muscle": "Back", "type": "isolation"},
    "Back Extensions":        {"movement": "hinge",           "met": 4.5, "muscle": "Back", "type": "isolation"},

    # --- LEGS ---
    "Squat (Barbell)":        {"movement": "squat", "met": 7.5, "muscle": "Legs", "type": "compound"},
    "Front Squat":            {"movement": "squat", "met": 7.5, "muscle": "Legs", "type": "compound"},
    "Goblet Squat":           {"movement": "squat", "met": 6.5, "muscle": "Legs", "type": "compound"},
    "Leg Press":              {"movement": "squat", "met": 6.0, "muscle": "Legs", "type": "compound"},
    "Lunges":                 {"movement": "lunge", "met": 7.0, "muscle": "Legs", "type": "compound"},
    "Bulgarian Split Squat":  {"movement": "lunge", "met": 7.5, "muscle": "Legs", "type": "compound"},
    "Leg Extension":          {"movement": "isolation", "met": 4.0, "muscle": "Legs", "type": "isolation"},
    "Hack Squat":             {"movement": "squat", "met": 6.5, "muscle": "Legs", "type": "compound"},
    "Deadlift (Conventional)":{"movement": "hinge", "met": 8.0, "muscle": "Legs", "type": "compound"},
    "Romanian Deadlift (RDL)":{"movement": "hinge", "met": 7.0, "muscle": "Legs", "type": "compound"},
    "Sumo Deadlift":          {"movement": "hinge", "met": 8.0, "muscle": "Legs", "type": "compound"},
    "Leg Curl (Seated/Lying)":{"movement": "isolation", "met": 4.0, "muscle": "Legs", "type": "isolation"},
    "Calf Raises (Standing)": {"movement": "isolation", "met": 3.5, "muscle": "Legs", "type": "isolation"},
    "Calf Raises (Seated)":   {"movement": "isolation", "met": 3.0, "muscle": "Legs", "type": "isolation"},

    # --- SHOULDERS ---
    "Overhead Press (OHP)":   {"movement": "vertical_push", "met": 6.5, "muscle": "Shoulders", "type": "compound"},
    "Dumbbell Shoulder Press":{"movement": "vertical_push", "met": 6.0, "muscle": "Shoulders", "type": "compound"},
    "Arnold Press":           {"movement": "vertical_push", "met": 6.0, "muscle": "Shoulders", "type": "compound"},
    "Lateral Raises":         {"movement": "isolation",     "met": 3.5, "muscle": "Shoulders", "type": "isolation"},
    "Front Raises":           {"movement": "isolation",     "met": 3.5, "muscle": "Shoulders", "type": "isolation"},
    "Rear Delt Fly":          {"movement": "isolation",     "met": 3.5, "muscle": "Shoulders", "type": "isolation"},
    "Upright Row":            {"movement": "vertical_pull", "met": 5.0, "muscle": "Shoulders", "type": "compound"},

    # --- ARMS ---
    "Barbell Curl":           {"movement": "isolation", "met": 4.0, "muscle": "Arms", "type": "isolation"},
    "Dumbbell Curl":          {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Hammer Curl":            {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Preacher Curl":          {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Cable Bicep Curl":       {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Tricep Pushdown":        {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Skullcrushers":          {"movement": "isolation", "met": 4.0, "muscle": "Arms", "type": "isolation"},
    "Overhead Tricep Ext":    {"movement": "isolation", "met": 3.5, "muscle": "Arms", "type": "isolation"},
    "Close Grip Bench Press": {"movement": "horizontal_push", "met": 5.5, "muscle": "Arms", "type": "compound"},
    "Dips (Tricep Focus)":    {"movement": "vertical_push", "met": 6.0, "muscle": "Arms", "type": "compound"},

    # --- CORE ---
    "Plank":                  {"movement": "stability", "met": 4.5, "muscle": "Core", "type": "isolation"},
    "Crunches":               {"movement": "flexion",   "met": 3.5, "muscle": "Core", "type": "isolation"},
    "Leg Raises":             {"movement": "flexion",   "met": 4.0, "muscle": "Core", "type": "isolation"},
    "Russian Twists":         {"movement": "rotation",  "met": 4.5, "muscle": "Core", "type": "isolation"},
    "Ab Wheel Rollout":       {"movement": "extension", "met": 5.5, "muscle": "Core", "type": "compound"},

    # --- CARDIO ---
    "Burpees":                {"movement": "plyometric", "met": 9.5, "muscle": "Full Body", "type": "cardio"},
    "Jumping Jacks":          {"movement": "cardio",     "met": 8.0, "muscle": "Full Body", "type": "cardio"},
    "Mountain Climbers":      {"movement": "cardio",     "met": 8.5, "muscle": "Full Body", "type": "cardio"},
    "Box Jumps":              {"movement": "plyometric", "met": 8.0, "muscle": "Full Body", "type": "cardio"},
    "Kettlebell Swing":       {"movement": "hinge",      "met": 7.5, "muscle": "Full Body", "type": "compound"}
}


# --- NEW: AI PREDICTION ENGINE (Regression) ---
def predict_next_weight(username, exercise_name):
    workouts = load_data(WORKOUT_FILE).get(username, [])
    
    data_points = []
    
    for w in workouts:
        if w["exercise"] == exercise_name and w.get("weight") != "-":
            try:
                # Handle "50, 60" -> take max weight
                weight_str = str(w["weight"])
                weights = [float(x.strip()) for x in weight_str.split(',') if x.strip().replace('.','',1).isdigit()]
                if not weights: continue
                best_weight = max(weights)
                
                reps = int(w["reps"])
                one_rm = best_weight * (1 + reps/30)
                
                dt = datetime.strptime(w["date"], "%Y-%m-%d")
                timestamp = dt.toordinal()
                data_points.append((timestamp, one_rm))
            except:
                continue
                
    if len(data_points) < 3:
        return None

    data_points.sort(key=lambda x: x[0])
    
    # Linear Regression (Least Squares)
    n = len(data_points)
    sum_x = sum(p[0] for p in data_points)
    sum_y = sum(p[1] for p in data_points)
    sum_xy = sum(p[0] * p[1] for p in data_points)
    sum_xx = sum(p[0] * p[0] for p in data_points)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0: return None
    
    m = (n * sum_xy - sum_x * sum_y) / denominator
    c = (sum_y - m * sum_x) / n
    
    today_ordinal = datetime.today().toordinal()
    predicted_1rm = m * today_ordinal + c
    suggested_working_weight = round(predicted_1rm * 0.75, 1)
    
    trend = "📈 Improving" if m > 0 else "📉 Plateau"
    
    return {
        "predicted_1rm": round(predicted_1rm, 1),
        "suggested_weight": suggested_working_weight,
        "trend": trend
    }


def calculate_calories(workout_list, user_weight):
    total = 0
    details = []

    for item in workout_list:
        name = item["exercise"]
        sets = int(item["sets"])
        reps = int(item["reps"])

        if name not in EXERCISES:
            continue

        met = EXERCISES[name]["met"]
        total_reps = sets * reps
        minutes = (total_reps * 3) / 60
        cal_per_min = met * 3.5 * float(user_weight) / 200
        burned = cal_per_min * minutes
        total += burned
        details.append((name, round(burned, 2)))

    return round(total, 2), details


def generate_ai_suggestions(workout_list):
    movement_volume = {}
    muscle_volume = {}
    compound_count = 0
    isolation_count = 0
    total_reps = 0

    for item in workout_list:
        exercise = item["exercise"]
        sets = int(item["sets"])
        reps = int(item["reps"])

        if exercise not in EXERCISES: continue

        info = EXERCISES[exercise]
        movement = info["movement"]
        muscle = info["muscle"]
        ex_type = info["type"]

        volume = sets * reps
        total_reps += volume

        movement_volume[movement] = movement_volume.get(movement, 0) + volume
        muscle_volume[muscle] = muscle_volume.get(muscle, 0) + volume

        if ex_type == "compound": compound_count += 1
        elif ex_type == "isolation": isolation_count += 1

    suggestions = []
    for movement, vol in movement_volume.items():
        if vol >= 100:
            suggestions.append(f"You did a lot of '{movement}' work today. Consider reducing similar exercises next time.")

    if muscle_volume.get("Legs", 0) == 0:
        suggestions.append("No leg exercises in this workout. Make sure you train legs at least once or twice a week.")

    if total_reps < 40:
        suggestions.append("Total volume was quite low. You can add a few more sets.")
    
    return suggestions
# 🔥 STREAK + BADGE SYSTEM
def calculate_streak_and_badges(username):
    workouts = load_data(WORKOUT_FILE).get(username, [])

    if not workouts:
        return 0, "No Badge Yet"

    # Get unique workout dates
    dates = sorted(list(set([w["date"] for w in workouts])))

    from datetime import datetime, timedelta

    # Convert to datetime
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

    # 🔥 STREAK CALCULATION
    streak = 0
    today = datetime.today()

    for i in range(len(dates)-1, -1, -1):
        if dates[i].date() == (today - timedelta(days=streak)).date():
            streak += 1
        else:
            break

    # 🔥 MONTHLY CONSISTENCY
    current_month = today.month
    current_year = today.year

    monthly_days = len(set([
        d.strftime("%Y-%m-%d")
        for d in dates
        if d.month == current_month and d.year == current_year
    ]))

    # 🏆 BADGES
    if monthly_days >= 25:
        badge = "🏆 Elite Athlete"
    elif monthly_days >= 20:
        badge = "🔥 Consistency King"
    elif monthly_days >= 15:
        badge = "💪 Dedicated"
    elif monthly_days >= 10:
        badge = "👍 Getting There"
    else:
        badge = "🚀 Beginner"

    return streak, badge

# ----------------- Routes -----------------
# 🔥 GRAPH DATA API (ADD THIS)
@app.route('/get_graph_data')
def get_graph_data():
    username = request.args.get("username")
    range_type = request.args.get("range")

    data = load_data(WORKOUT_FILE)
    user_workouts = data.get(username, [])

    from collections import defaultdict

    result = defaultdict(float)
    today = datetime.today()

    for w in user_workouts:
        try:
            date = datetime.strptime(w["date"], "%Y-%m-%d")

            # 🔥 CALORIES CALCULATION (better than count)
            exercise = w["exercise"]
            if exercise in EXERCISES:
                met = EXERCISES[exercise]["met"]
                sets = int(w["sets"])
                reps = int(str(w["reps"]).split("-")[0])

                minutes = (sets * reps * 3) / 60
                burned = met * 3.5 * 70 / 200 * minutes   # approx weight

            else:
                burned = 1

            # --- FILTER BASED ON RANGE ---
            if range_type == "week":
                if date >= today - timedelta(days=7):
                    key = date.strftime("%a")  # Mon, Tue
                    result[key] += burned

            elif range_type == "month":
                if date >= today - timedelta(days=30):
                    key = date.strftime("%d %b")  # 12 Mar
                    result[key] += burned

            elif range_type == "year":
                if date.year == today.year:
                    key = date.strftime("%b")  # Jan, Feb
                    result[key] += burned

        except:
            continue

    # 🔥 SORT DATA (IMPORTANT)
    sorted_data = sorted(result.items(), key=lambda x: x[0])

    labels = [x[0] for x in sorted_data]
    values = [round(x[1], 1) for x in sorted_data]

    return jsonify({
        "labels": labels,
        "data": values
    })
# --- NEW: GEMINI MONTHLY REPORT ROUTE ---
@app.route("/analyze_month", methods=["POST"])
def analyze_month():
    if not ai_model:
        return jsonify({"response": "AI not initialized. Check API Key."})

    data = request.json
    username = data.get("username")
    year = int(data.get("year"))
    month = int(data.get("month")) + 1 

    workouts = load_data(WORKOUT_FILE).get(username, [])
    
    month_workouts = []
    for w in workouts:
        try:
            w_date = datetime.strptime(w["date"], "%Y-%m-%d")
            if w_date.year == year and w_date.month == month:
                month_workouts.append(w)
        except:
            continue

    if not month_workouts:
        return jsonify({"response": "No workouts found for this month! Go lift some weights first. 💪"})

    workout_text = json.dumps(month_workouts, indent=2)
    prompt = f"""
    Act as a tough but encouraging professional gym coach. 
    Here is my workout history for this month (JSON format):
    {workout_text}

    Please provide a concise monthly review (max 150 words) covering:
    1. Consistency.
    2. Volume/Intensity highlights.
    3. One specific critique.
    
    Use emojis. Address me directly.
    """

    try:
        response = ai_model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"AI Error: {str(e)}"})


# --- NEW: NLP WORKOUT PARSER ---
@app.route("/parse_workout_text", methods=["POST"])
def parse_workout_text():
    if not ai_model:
        return jsonify({"error": "AI not initialized. Check API Key."})

    data = request.json
    text = data.get("text")
    if not text: return jsonify({"error": "No text provided"})

    prompt = f"""
    Extract workout data from this text: "{text}"
    Return ONLY a raw JSON list. No markdown formatting.
    Match exercises to this list if possible: {list(EXERCISES.keys())}
    Format: [{{"exercise": "Name", "sets": "num", "reps": "num", "weight": "num"}}]
    If weight unit missing, assume kg.
    """
    
    try:
        response = ai_model.generate_content(prompt)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(cleaned_text)
        return jsonify(parsed_data)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/register_user", methods=["POST"])
def register_user():
    users = load_data(USERS_FILE)
    username = request.form["username"]
    if username in users:
        return "User already exists!"

    hashed_pw = generate_password_hash(request.form["password"],method='pbkdf2:sha256')
    users[username] = {
        "password": hashed_pw,
        "name": request.form["name"],
        "age": request.form["age"],
        "weight": request.form["weight"]
    }
    save_data(USERS_FILE, users)
    return redirect("/")

@app.route("/login_user", methods=["POST"])
def login_user():
    users = load_data(USERS_FILE)
    username = request.form["username"]
    password = request.form["password"]

    if username in users and check_password_hash(users[username]["password"], password):
        return redirect(f"/dashboard/{username}")
    else:
        return "Invalid login! Try Again"

@app.route("/dashboard/<username>")
def dashboard(username):
    users = load_data(USERS_FILE)
    user = users.get(username)
    if not user:
        return "User not found", 404

    workouts = load_data(WORKOUT_FILE).get(username, [])

    # 🔥 1. Chart Data
    daily_cals = {}
    for w in workouts:
        date_str = w.get("date")
        if not date_str:
            continue

        exercise = w["exercise"].strip()
        if exercise in EXERCISES:
            met = EXERCISES[exercise]["met"]
            sets = int(w["sets"])
            try:
                reps = int(str(w["reps"]).split("-")[0])
            except:
                reps = 8

            minutes = (sets * reps * 3) / 60
            burned = met * 3.5 * float(user["weight"]) / 200 * minutes
            daily_cals[date_str] = daily_cals.get(date_str, 0) + burned

    dates_labels = []
    calories_data = []
    today = datetime.today()

    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        d_str = day.strftime("%Y-%m-%d")

        dates_labels.append(day.strftime("%b %d"))
        calories_data.append(round(daily_cals.get(d_str, 0), 1))

    # 🔥 2. PRs
    prs = {
        "Bench Press": 0,
        "Squat (Barbell)": 0,
        "Deadlift (Conventional)": 0
    }

    for w in workouts:
        name = w["exercise"]
        weight_str = w.get("weight", "0")

        if name in prs:
            try:
                values = [
                    float(x.strip())
                    for x in weight_str.split(',')
                    if x.strip().replace('.', '', 1).isdigit()
                ]

                if values:
                    max_lift = max(values)
                    if max_lift > prs[name]:
                        prs[name] = max_lift
            except:
                pass

    # 🔥 3. Predictions
    predictions = {}
    target_lifts = [
        "Bench Press",
        "Squat (Barbell)",
        "Deadlift (Conventional)"
    ]

    for lift in target_lifts:
        pred = predict_next_weight(username, lift)
        if pred:
            predictions[lift] = pred

    # 🔥 4. STREAK + BADGE (NEW 🔥🔥🔥)
    streak, badge = calculate_streak_and_badges(username)

    print("STREAK:", streak)
    print("BADGE:", badge)

    # 🔥 FINAL RETURN
    return render_template(
        "dashboard.html",
        user=user,
        username=username,
        dates_labels=dates_labels,
        calories_data=calories_data,
        prs=prs,
        predictions=predictions,
        streak=streak,
        badge=badge
    )
@app.route("/history/<username>")
def workout_history(username):
    workouts = load_data(WORKOUT_FILE).get(username, [])
    return render_template("history.html", workouts=workouts, username=username)

@app.route("/workout/<username>")
def workout_page(username):
    exercise_groups = {}
    for name, info in EXERCISES.items():
        muscle = info["muscle"]
        if muscle not in exercise_groups: exercise_groups[muscle] = []
        exercise_groups[muscle].append(name)

    for m in exercise_groups: exercise_groups[m].sort()

    return render_template("workout.html", username=username, exercise_groups=exercise_groups)

@app.route("/save_workout/<username>", methods=["POST"])
def save_workout(username):
    workouts = load_data(WORKOUT_FILE)
    users = load_data(USERS_FILE)
    if username not in workouts: workouts[username] = []
    today = datetime.today().strftime("%Y-%m-%d")

    exercises = request.form.getlist("exercise[]")
    sets_list = request.form.getlist("sets[]")
    reps_list = request.form.getlist("reps[]")
    weights_list = request.form.getlist("weight[]") 

    for i in range(len(exercises)):
        if i < len(sets_list) and i < len(reps_list):
            entry = {
                "date": today,
                "exercise": exercises[i],
                "sets": sets_list[i],
                "reps": reps_list[i],
                "weight": weights_list[i] if i < len(weights_list) else "-" 
            }
            workouts[username].append(entry)

    save_data(WORKOUT_FILE, workouts)
    
    today_workouts = [w for w in workouts[username] if w.get("date") == today]
    user_weight = users.get(username, {}).get("weight", 70) 
    total, details = calculate_calories(today_workouts, user_weight)
    suggestions = generate_ai_suggestions(today_workouts)

    return render_template("summary.html", total=total, details=details, suggestions=suggestions, username=username)

@app.route("/delete_workout/<username>/<int:index>")
def delete_workout(username, index):
    workouts = load_data(WORKOUT_FILE)
    user_workouts = workouts.get(username, [])
    if 0 <= index < len(user_workouts):
        user_workouts.pop(index) 
        save_data(WORKOUT_FILE, workouts)
    return redirect(f"/history/{username}")

@app.route("/settings/<username>")
def settings_page(username):
    users = load_data(USERS_FILE)
    user = users.get(username)
    if not user: return "User not found", 404
    return render_template("settings.html", user=user, username=username)

@app.route("/update_settings/<username>", methods=["POST"])
def update_settings(username):
    users = load_data(USERS_FILE)
    if username not in users: return "User not found", 404
    
    users[username]["name"] = request.form["name"]
    users[username]["age"] = request.form["age"]
    users[username]["weight"] = request.form["weight"]

    new_password = request.form["password"]
    if new_password and new_password.strip() != "":
        users[username]["password"] = generate_password_hash(new_password)

    save_data(USERS_FILE, users)
    return redirect(f"/dashboard/{username}")

# --- ADD THESE ROUTES TO YOUR app.py ---

@app.route("/planner/<username>")
def planner_page(username):
    users = load_data(USERS_FILE) #
    user = users.get(username)
    if not user: return "User not found", 404
    return render_template("planner.html", username=username, user=user)

@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    goal_raw = data.get("goal")
    diet_raw = data.get("diet_pref")

    # Load user stats
    users = load_data(USERS_FILE)
    user = users.get(username, {})
    try:
        weight = float(user.get("weight", 70))
    except (ValueError, TypeError):
        weight = 70.0

    # 1. Feature Encoding (Translating dropdowns to ML numbers)
    goal_map = {"Weight Loss": 0, "Weight Gain": 1, "Maintenance": 2}
    diet_map = {"Balanced": 0, "Vegetarian": 1, "High Protein": 2, "Vegan": 3}

    goal_encoded = goal_map.get(goal_raw, 2)
    diet_encoded = diet_map.get(diet_raw, 0)
    weight_cat = 0 if weight < 65 else (1 if weight <= 80 else 2)

    # 2. Local ML Prediction
    try:
        prediction = ml_workout_planner.predict([[goal_encoded, diet_encoded, weight_cat]])[0]
        # Pass the encoded features to the generator so KNN can find the exact right meals
        html_plan = get_ml_plan_template(prediction, goal_encoded, diet_encoded) 
        return jsonify({"plan": html_plan})
    except Exception as e:
        return jsonify({"error": f"ML Prediction failed: {str(e)}"})

# --- MR. OLYMPIA FEATURE ROUTES ---

@app.route("/olympia/<username>")
def olympia_page(username):
    users = load_data(USERS_FILE)
    if username not in users: return "User not found", 404
    return render_template("olympia.html", username=username)

@app.route("/generate_olympia_plan", methods=["POST"])
def generate_olympia_plan():
    if not ai_model:
        return jsonify({"error": "AI not initialized"})

    data = request.json
    stats = data.get("stats")
    
    # Prompt engineering for specialized bodybuilding advice
    prompt = f"""
    Act as a strict IFBB Pro Bodybuilding Coach.
    My current stats:
    - Chest: {stats['chest']}"
    - Arms: {stats['arms']}"
    - Waist: {stats['waist']}"
    - Thighs: {stats['thighs']}"
    - Calves: {stats['calves']}"

    The Goal: Reach "Classic Physique" standards (Golden Era proportions).
    
    Provide a response in HTML format (no markdown, just <h3>, <p>, <ul>):
    1. **Critique**: Brutally honest comparison. Which body part is lagging the most?
    2. **The Fix (Workout)**: A specialized 4-day split emphasizing the weak points.
    3. **The Fuel (Diet)**: A specific macro strategy to grow the weak areas without getting a "bloated gut".
    
    Keep the tone motivating but hardcore. Use emojis like 🏆, 💪, 🍗.
    """
    
    try:
        response = ai_model.generate_content(prompt)
        return jsonify({"plan": response.text})
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route("/pain_recovery", methods=["POST"])
def pain_recovery():
    data = request.json
    
    muscle = data.get("muscle")
    days = int(data.get("days"))
    pain = int(data.get("pain"))

    # --- Basic logic ---
    if days <= 2 and pain <= 5:
        status = "Normal DOMS"
        advice = "Light movement, hydration, and stretching."

    elif days <= 4 and pain <= 7:
        status = "Delayed Recovery"
        advice = "Take rest, avoid heavy training, do light stretching."

    else:
        status = "Possible Injury"
        advice = "Stop training this muscle and consider medical check."

    # --- AI Enhancement (optional) ---
    if ai_model:
        try:
            prompt = f"""
            A user has {muscle} pain for {days} days with pain level {pain}/10.

            Give short recovery advice:
            - Is it normal or injury?
            - Should they train or rest?
            - 3 tips
            """
            response = ai_model.generate_content(prompt)
            return jsonify({"response": response.text})
        except:
            pass

    return jsonify({"response": f"{status}\n{advice}"})










if __name__ == "__main__":
    app.run(debug=True, port=5001)