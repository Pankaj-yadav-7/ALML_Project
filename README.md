# AI Fitness & Diet Suggester

A beginner-friendly AI + Machine Learning project that provides personalized workout and diet recommendations based on user inputs like age, weight, height, gender, and fitness goals.

Built as a first-year B.Tech (CSE in AI & ML) capstone project.

---

## Features

- ML-based workout recommendation using KNN
- Suggests workout types:
  - Cardio
  - Weightlifting
  - Yoga
  - HIIT
- Goal-based diet plans:
  - Weight Loss (Calorie Deficit)
  - Muscle Gain (Calorie Surplus)
- Health snapshot including:
  - BMI calculation
  - Daily water intake
  - Calorie estimation (TDEE)
- Session logging using CSV
- View and clear past recommendations
- Input validation to prevent crashes

---

## Tech Stack

- Language: Python 3
- Libraries:
  - pandas
  - scikit-learn
  - os
  - warnings
- Interface: Command Line (CLI)
- Storage: CSV files

---

## Machine Learning Approach

- Model: K-Nearest Neighbours (KNN)
- Features used:
  - Age
  - Weight
  - Goal (encoded)
- Output:
  - Workout type recommendation

KNN works by finding similar users in the dataset and recommending a workout based on nearest matches.

---

## Project Structure

```
AI_fitness_suggester.py
fitness_dataset.csv
user_recommendation_logs.csv
fitness_project_report.docx
README.md
```

---

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-fitness-suggester.git
   cd ai-fitness-suggester
   ```

2. Install dependencies:
   ```
   pip install pandas scikit-learn
   ```

3. Run the program:
   ```
   python "AI_fitness_suggester.py"
   ```

---

## How It Works

1. User inputs:
   - Age
   - Weight
   - Height
   - Gender
   - Goal

2. System:
   - Calculates BMI, water intake, and calories
   - Runs the ML model
   - Generates workout and diet plan

3. Output:
   - Weekly workout routine
   - Nutrition plan
   - Health insights

---

## Sample Output

```
BMI: 23.4 (Normal weight)
Water Intake: 2.6 liters
Calories: ~2200 kcal/day

Recommended Workout: Weightlifting
Recommended Diet: Muscle Gain Plan
```

---

## Limitations

- Very small dataset (12 rows)
- Limited workout categories
- No fitness level input
- CLI-based (no graphical interface)

---

## Future Improvements

- Larger dataset for better accuracy
- Add fitness level input (beginner/intermediate/advanced)
- Build GUI using Tkinter or web framework
- Add more workout categories
- Analyze user logs for insights

---

## Author

Pankaj Yadav  
B.Tech (AI & ML) – 1st Year  

---

## Acknowledgment

This project was developed as part of the Fundamentals in AI & ML course.
