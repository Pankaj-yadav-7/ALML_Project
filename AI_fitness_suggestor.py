import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
import warnings

# sklearn keeps screaming warnings, just make it stop
warnings.filterwarnings("ignore")

DATA_FILE = 'fitness_dataset.csv'
LOG_FILE = 'user_recommendation_logs.csv'

# Map goal names to numeric codes the model understands
GOAL_MAP = {
    'weight loss': 0,
    'muscle gain': 1
}

WORKOUT_ROUTINES = {
    'Cardio': """
    Weekly Cardio & Endurance Plan:
    - Monday: 30 min steady-state jogging or brisk walking.
    - Tuesday: 20 min cycling or swimming.
    - Wednesday: Active rest (light stretching or 15 min walk).
    - Thursday: 30 min interval running (1 min fast, 2 min slow).
    - Friday: 45 min brisk walk or hiking.
    - Weekend: Rest or recreational sports.
    """,
    'Weightlifting': """
    Weekly Muscle Gain (Push/Pull/Legs) Plan:
    - Monday: Push Day (Bench press, overhead press, triceps extensions).
    - Tuesday: Pull Day (Barbell rows, pull-ups/lat pulldowns, bicep curls).
    - Wednesday: Rest and recovery.
    - Thursday: Leg Day (Squats, lunges, Romanian deadlifts, calf raises).
    - Friday: Full Body or weak-point focus (light weight, high reps).
    - Weekend: Rest and light walking.
    """,
    'Yoga': """
    Weekly Flexibility & Core (Yoga) Plan:
    - Monday: 45 min Vinyasa flow (full body mobility).
    - Tuesday: 30 min core-focused Pilates or Yoga.
    - Wednesday: Rest.
    - Thursday: 45 min Hatha Yoga (focusing on holding poses).
    - Friday: 30 min Restorative or Yin Yoga for deep stretching.
    - Weekend: Active rest.
    """,
    'HIIT': """
    Weekly High-Intensity Interval Training (HIIT) Plan:
    - Monday: 20 min Bodyweight HIIT (30s work, 30s rest: burpees, squats).
    - Tuesday: Active recovery (light stretching or walking).
    - Wednesday: 25 min Dumbbell HIIT or Kettlebell swings.
    - Thursday: Rest.
    - Friday: 20 min Sprint intervals (on a track, bike, or treadmill).
    - Weekend: Rest.
    """
}

DIET_PLANS = {
    'weight loss': """
    Nutrition Plan: Caloric Deficit & High Satiety Focus
    - Breakfast: Oatmeal with berries or a 3-egg-white spinach omelet.
    - Lunch: Grilled chicken salad with a light vinaigrette.
    - Snack: Greek yogurt or a small handful of almonds.
    - Dinner: Baked salmon or tofu with roasted asparagus and a small portion of quinoa.
    - Hydration: Minimum 3 liters of water daily. Replace sugary drinks with black coffee or green tea.
    """,
    'muscle gain': """
    Nutrition Plan: Caloric Surplus & High Protein Focus
    - Breakfast: 3 scrambled whole eggs, 2 slices of whole-wheat toast, and a protein shake.
    - Lunch: Large portion of chicken breast, brown rice, and steamed broccoli.
    - Snack: Peanut butter on rice cakes or a bowl of cottage cheese.
    - Dinner: Lean steak or ground turkey, sweet potato, and mixed vegetables.
    - Post-Workout: Whey protein isolate and a banana.
    """
}

# one tip per workout per gender, shown at the end of the plan
GENDER_WORKOUT_TIPS = {
    'male': {
        'Cardio': 'Focus on maintaining a strong pace. Track your heart rate to stay in the fat-burn zone (60-70% max HR).',
        'Weightlifting': 'Progressive overload is key — try to add small weight or an extra rep each week.',
        'Yoga': 'Don\'t skip the strength-focused poses like Warrior and Chair — they build functional muscle.',
        'HIIT': 'Push hard on the work intervals. Your recovery will improve fast with consistency.',
    },
    'female': {
        'Cardio': 'Mix in incline walking — it targets glutes and burns more calories than flat jogging.',
        'Weightlifting': 'Don\'t fear heavier weights. Lifting heavy won\'t bulk you up — it\'ll tone and strengthen.',
        'Yoga': 'Hip-opening poses like Pigeon and Lizard are especially beneficial. Prioritise them.',
        'HIIT': 'Lower-body focused HIIT (glute bridges, jump squats) will complement your goals well.',
    }
}


def initialize_dataset():
    # Create a starter dataset if one doesn't exist yet.
    if os.path.exists(DATA_FILE):
        return

    print("Setting things up for the first time — this will only take a moment!")

    data = {
        'age': [18, 22, 25, 30, 45, 50, 20, 23, 35, 40, 28, 55],
        'weight_kg': [60, 85, 70, 95, 65, 90, 55, 100, 80, 75, 68, 88],
        'goal_code': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        'workout_type': [
            'Cardio', 'Weightlifting', 'Cardio', 'Weightlifting',
            'Yoga', 'Weightlifting', 'Cardio', 'Weightlifting',
            'HIIT', 'Weightlifting', 'Cardio', 'Yoga'
        ]
    }

    pd.DataFrame(data).to_csv(DATA_FILE, index=False)
    print("All set! Let's get started.\n")


def train_model():
    # Load the dataset and train a KNN classifier to predict workout type.
    try:
        df = pd.read_csv(DATA_FILE)
        X = df[['age', 'weight_kg', 'goal_code']]
        y = df['workout_type']

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        return model

    except Exception as e:
        print(f"Oops! Something went wrong while loading the fitness model: {e}")
        return None


# bmi 

def calculate_bmi(weight, height_m):
    # Standard BMI formula: weight (kg) / height (m)^2
    return round(weight / (height_m ** 2), 1)

def get_bmi_category(bmi):
    # Return a plain-English label for the BMI value
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"

def show_bmi(weight, height_m):
    bmi = calculate_bmi(weight, height_m)
    category = get_bmi_category(bmi)

    notes = {
        "Underweight":   "Consider increasing calorie intake with nutrient-dense foods.",
        "Normal weight": "Great foundation to work with! Focus on your goal.",
        "Overweight":    "A mix of cardio and strength work will help a lot.",
        "Obese":         "Start with low-impact exercise and consult a doctor if unsure."
    }

    print(f"\n  BMI: {bmi}  ({category})")
    print(f"  Note: {notes[category]}")


# water intake

def calculate_water_intake(weight):
    # I looked up like 6 different sources and they all said something different, going with 35ml/kg
    liters = round((weight * 35) / 1000, 1)
    return liters

def show_water_intake(weight):
    liters = calculate_water_intake(weight)
    print(f"\n  Daily water intake target: {liters} liters")
    print("  Tip: Spread it across the day — don't try to drink it all at once!")


# calorie estimate using TDEE

def calculate_tdee(age, weight, height_m, goal, gender):
    # Mifflin-St Jeor: male constant is +5, female is -161. Now that we ask gender, use it properly.
    gender_constant = 5 if gender == 'male' else -161
    bmr = (10 * weight) + (6.25 * height_m * 100) - (5 * age) + gender_constant
    tdee = round(bmr * 1.55)  # Moderate activity level

    # Adjust calories based on goal
    if goal == 'weight loss':
        target = tdee - 500   # ~0.5 kg loss per week
        label = "Caloric Deficit Target"
    else:
        target = tdee + 300   # Lean bulk
        label = "Caloric Surplus Target"

    return tdee, target, label

def show_calorie_estimate(age, weight, height_m, goal, gender):
    tdee, target, label = calculate_tdee(age, weight, height_m, goal, gender)
    print(f"\n  Estimated maintenance calories: ~{tdee} kcal/day")
    print(f"  {label}: ~{target} kcal/day")


# input helpers — all of these just loop until the user gives something valid

def ask_for_age():
    # Keep asking until we get a valid age.
    while True:
        try:
            age = int(input("How old are you? (e.g., 21): "))
            if age > 0:
                return age
            print("Hmm, that doesn't seem right. Please enter a valid age.")
        except ValueError:
            print("Please enter a whole number for age.")


def ask_for_weight():
    # Keep asking until we get a valid weight.
    while True:
        try:
            weight = float(input("What's your current weight in kg? (e.g., 75.5): "))
            if weight > 0:
                return weight
            print("That doesn't look like a valid weight. Try again!")
        except ValueError:
            print("Please enter a number for weight (e.g., 70 or 68.5).")


def ask_for_height():
    # Keep asking until we get a valid height in cm, then convert to metres.
    while True:
        try:
            height_cm = float(input("What's your height in cm? (e.g., 175): "))
            if height_cm > 0:
                return height_cm / 100  # store as metres for BMI formula
            print("That doesn't look right. Please enter your height in centimetres.")
        except ValueError:
            print("Please enter a number for height (e.g., 170).")


def ask_for_goal():
    # Keep asking until the user picks a recognised goal.
    while True:
        goal = input("What's your main goal? Type 'weight loss' or 'muscle gain': ").strip().lower()
        if goal in GOAL_MAP:
            return goal
        print("Please type exactly 'weight loss' or 'muscle gain'.")


def ask_for_gender():
    # Keep asking until we get male or female.
    while True:
        gender = input("What's your gender? Type 'male' or 'female': ").strip().lower()
        if gender in ('male', 'female'):
            return gender
        print("Please type 'male' or 'female'.")


def get_user_profile():
    # Collect age, weight, height, gender, and goal from the user.
    age = ask_for_age()
    weight = ask_for_weight()
    height_m = ask_for_height()
    gender = ask_for_gender()
    goal = ask_for_goal()
    return age, weight, height_m, gender, goal


# display and logging

def show_recommendation(prediction, goal, gender):
    # Print the recommended workout and diet plan in a readable format.
    print("\n Crunching the numbers... Here's what we recommend for you!")

    print(f"\n Your Recommended Workout Style: {prediction}")
    print("-" * 50)
    print(WORKOUT_ROUTINES.get(prediction, "No detailed plan available for this workout type yet."))

    # Append a short gender-specific tip below the main plan
    tip = GENDER_WORKOUT_TIPS.get(gender, {}).get(prediction)
    if tip:
        print(f"  Tip for you: {tip}")

    print(f"\n Your Nutrition Approach: {goal.title()}")
    print("-" * 50)
    print(DIET_PLANS.get(goal, "No detailed diet plan available."))
    print("-" * 50)


def log_recommendation(age, weight, height_m, gender, goal, workout, diet_type, bmi):
    entry = pd.DataFrame({
        'Age': [age],
        'Weight_kg': [weight],
        'Height_m': [height_m],
        'BMI': [bmi],
        'Gender': [gender.title()],
        'Goal': [goal.title()],
        'Recommended_Workout': [workout],
        'Diet_Focus': [diet_type]
    })

    # If the file already exists, just append without repeating the header
    write_header = not os.path.exists(LOG_FILE)
    entry.to_csv(LOG_FILE, mode='a', header=write_header, index=False)


def view_logs():
    # Display all past recommendations.
    if not os.path.exists(LOG_FILE):
        print("\nNo previous logs found. Generate a recommendation first.\n")
        return

    print("\n Here's a look at your past recommendations:\n")
    logs = pd.read_csv(LOG_FILE)
    print(logs.to_string(index=False))
    print("\n(That's all of them so far!)\n")


def clear_logs():
    # Delete the log file after asking the user to confirm.
    if not os.path.exists(LOG_FILE):
        print("\nThere are no logs to clear.\n")
        return

    confirm = input("Are you sure you want to delete all logs? This can't be undone. (yes/no): ").strip().lower()
    # requiring full 'yes' not just 'y' — once deleted everything with a misclick and never recovered
    if confirm == 'yes':
        os.remove(LOG_FILE)
        print("Logs cleared successfully.\n")
    else:
        print("No changes made.\n")


def main():
    print("==========================================")
    print("      AI Fitness & Diet Suggester         ")
    print("==========================================")

    initialize_dataset()

    model = train_model()
    if model is None:
        print("Sorry, we ran into a problem loading the fitness model. Please try restarting the app.")
        return

    while True:
        print("\nWhat would you like to do?")
        print("  1.   Get a personalized fitness & diet recommendation")
        print("  2.   View my past recommendations")
        print("  3.   Clear all logs")
        print("  4.   Exit")

        choice = input("\nYour choice (1, 2, 3, or 4): ").strip()

        if choice == '1':
            print("\nGreat! Let's learn a little about you first.\n")

            age, weight, height_m, gender, goal = get_user_profile()

            # Show BMI, water intake, and calorie estimate before the workout plan
            print("\n" + "=" * 50)
            print("  Your Health Snapshot")
            print("=" * 50)
            show_bmi(weight, height_m)
            show_water_intake(weight)
            show_calorie_estimate(age, weight, height_m, goal, gender)
            print("=" * 50)

            # Convert goal to numeric so the model can use it
            goal_code = GOAL_MAP[goal]
            user_input = pd.DataFrame([[age, weight, goal_code]], columns=['age', 'weight_kg', 'goal_code'])

            prediction = model.predict(user_input)[0]

            show_recommendation(prediction, goal, gender)

            diet_summary = "Deficit" if goal == 'weight loss' else "Surplus"
            bmi = calculate_bmi(weight, height_m)
            log_recommendation(age, weight, height_m, gender, goal, prediction, diet_summary, bmi)
            print("\n Your recommendation has been saved! Keep up the great work!\n")

        elif choice == '2':
            view_logs()

        elif choice == '3':
            clear_logs()

        elif choice == '4':
            print("Shutting down system. Goodbye!")
            break

        else:
            print("Hmm, that's not a valid option. Please type 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()