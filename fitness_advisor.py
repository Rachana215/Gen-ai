import streamlit as st
import weaviate
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the Weaviate client
weaviate_client = weaviate.Client("http://localhost:8085")

# Initialize the GPT-J model and tokenizer
@st.cache_resource
def load_gptj_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    return tokenizer, model

tokenizer, model = load_gptj_model()

# Function to fetch exercise data based on a fitness goal
def fetch_exercise_data_gptj(fitness_goal):
    # Use GPT-J to encode the fitness goal
    inputs = tokenizer(fitness_goal, return_tensors="pt")
    with st.spinner("Processing your request..."):
        outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
    goal_embedding = outputs[0].tolist()

    query = {
        "vector": goal_embedding,
        "certainty": 0.5,  # Lower threshold for flexible matching
    }

    try:
        # Perform the query with nearVector
        results = weaviate_client.query.get(
            "Exercise", ["name", "description"]
        ).with_near_vector(query).do()

        # Extract exercise data
        exercises = results.get("data", {}).get("Get", {}).get("Exercise", [])
        if not exercises:  # Return fallback if no matches
            return [{"name": "Walking", "description": "A basic cardio activity for all fitness levels."}]

        return exercises

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return [{"name": "Error", "description": "An error occurred while retrieving data."}]

# Function to populate exercise data into Weaviate (if schema is empty)
def populate_exercise_data():
    try:
        existing_data = weaviate_client.query.get("Exercise", ["name"]).do()
        if existing_data.get("data", {}).get("Get", {}).get("Exercise", []):
            st.write("Data already exists in Weaviate.")
            return
    except Exception:
        st.write("Populating new data...")

    # Example exercise data
    exercises = [
        {"name": "Push-ups", "description": "A basic upper-body strength exercise."},
        {"name": "Squats", "description": "A lower-body strength exercise."},
        {"name": "Jogging", "description": "A cardio activity to improve endurance."},
    ]

    for exercise in exercises:
        weaviate_client.data_object.create(exercise, class_name="Exercise")

# Streamlit App Layout
st.title("Personalized Fitness and Diet Advisor (GPT-J Edition)")

# Input Section
fitness_goal = st.text_input("Enter your fitness goal (e.g., weight loss, muscle gain):")

if st.button("Generate Plan"):
    if not fitness_goal:
        st.error("Please enter a fitness goal.")
    else:
        # Fetch exercises
        exercises = fetch_exercise_data_gptj(fitness_goal)

        # Display Suggested Exercises
        st.subheader("Suggested Exercises")
        if exercises and exercises[0].get("name") != "Error":
            for exercise in exercises:
                st.write(f"- *{exercise['name']}*: {exercise['description']}")
        else:
            st.write("No matching exercises found.")

        # Dummy nutritional information (for illustration)
        st.subheader("Nutritional Information")
        st.write("- Protein: 112.64 calories, 25g protein")
        st.write("- Salad: 19.98 calories, 1.23g protein")

        # AI Recommendations
        st.subheader("AI Recommendations")
        st.write(f"Based on your goal to '{fitness_goal}', focus on exercises and a diet tailored to your preferences.")

# Populate Data Button (for development/testing purposes)
if st.sidebar.button("Populate Exercise Data"):
    populate_exercise_data()

