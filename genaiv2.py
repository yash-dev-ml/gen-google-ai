import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import random
import gtts
from io import BytesIO
import sys
from io import StringIO

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

class SortingAlgorithm:
    def __init__(self, name, concepts):
        self.name = name
        self.concepts = concepts
        self.understood_concepts = set()

class UserModel:
    def __init__(self):
        self.algorithms = {
            "bubble_sort": SortingAlgorithm("Bubble Sort", ["basic concept", "implementation", "time complexity", "space complexity", "best/worst cases"]),
            "insertion_sort": SortingAlgorithm("Insertion Sort", ["basic concept", "implementation", "time complexity", "space complexity", "best/worst cases"]),
            "merge_sort": SortingAlgorithm("Merge Sort", ["basic concept", "implementation", "time complexity", "space complexity", "divide and conquer"]),
            "quick_sort": SortingAlgorithm("Quick Sort", ["basic concept", "implementation", "time complexity", "space complexity", "partitioning", "choosing pivot"])
        }
        self.current_algorithm = None
        self.current_concept = None
        self.conversation_history = []
        self.code_implementations = {}

    def set_current_algorithm(self, algorithm_name):
        self.current_algorithm = self.algorithms[algorithm_name]
        self.current_concept = next(iter(set(self.current_algorithm.concepts) - self.current_algorithm.understood_concepts), None)

    def update_understanding(self, concept):
        self.current_algorithm.understood_concepts.add(concept)
        self.current_concept = next(iter(set(self.current_algorithm.concepts) - self.current_algorithm.understood_concepts), None)

    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def get_next_algorithm(self):
        unlearned_algorithms = [alg for alg, obj in self.algorithms.items() if len(obj.understood_concepts) < len(obj.concepts)]
        return random.choice(unlearned_algorithms) if unlearned_algorithms else None

    def add_code_implementation(self, algorithm_name, code):
        self.code_implementations[algorithm_name] = code

@st.cache_resource
def get_user_models():
    return {}

def get_socratic_response(student_input, user_model):
    conversation = "\n".join([f"{entry['role']}: {entry['content']}" for entry in user_model.conversation_history])

    prompt = f"""
    You are an AI teaching assistant specializing in sorting algorithms. Your goal is to guide the student to understand and master sorting algorithms through Socratic questioning, clear explanations, and practical Python implementations when necessary.

    Current algorithm: {user_model.current_algorithm.name}
    Current concept: {user_model.current_concept}
    Understood concepts: {', '.join(user_model.current_algorithm.understood_concepts)}

    Recent conversation:
    {conversation}

    Student input: {student_input}

    Based on the student's progress and the conversation history:
    1. If introducing a new concept, provide a brief explanation followed by a thought-provoking question.
    2. If the student shows understanding, ask a more challenging question or present a scenario to apply the concept.
    3. If the student seems confused, break down the concept and ask simpler questions to build understanding.
    4. Regularly check for understanding and provide encouragement.
    5. If all concepts for the current algorithm are understood, prepare a short test question to confirm mastery.
    6. When appropriate, provide a Python code snippet demonstrating the concept or algorithm implementation.

    Respond in the voice of a supportive teacher, providing guidance and asking questions to promote critical thinking and practical implementation.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error("Oops! I'm having a bit of trouble thinking right now. Let's try a different approach!")
        return "I apologize for the confusion. Could you tell me what specific aspect of the algorithm you'd like to explore?"

def assess_understanding(student_input, assistant_response, user_model):
    prompt = f"""
    Analyze the following student response and determine if it demonstrates understanding of the current concept ({user_model.current_concept}) for the algorithm ({user_model.current_algorithm.name}).

    Student response: {student_input}
    Assistant's last question/guidance: {assistant_response}

    Provide a JSON response with the following structure:
    {{
        "understood": true/false,
        "confidence": 0-1 (float),
        "reasoning": "Brief explanation of your assessment"
    }}
    """

    try:
        response = model.generate_content(prompt)
        assessment = json.loads(response.text)
        
        if assessment["understood"] and assessment["confidence"] > 0.8:
            user_model.update_understanding(user_model.current_concept)
        
        return assessment["understood"], assessment.get("reasoning", "")
    except json.JSONDecodeError:
        st.info("I'm not quite sure if you've fully grasped this concept yet. Let's explore it a bit more!")
        return False, "Need more information to assess understanding."
    except Exception as e:
        st.error("I'm having trouble assessing your understanding. Let's continue our discussion!")
        return False, "Unable to assess understanding due to an error."

def generate_test_question(algorithm):
    prompt = f"""
    Create a comprehensive test question for the {algorithm.name} algorithm. The question should:
    1. Cover key concepts including {', '.join(algorithm.concepts)}
    2. Require application of knowledge, not just recall
    3. Be suitable for a short answer or code implementation

    Provide the question followed by key points that should be included in a correct answer.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error("I'm having trouble creating a test question. Let's review the algorithm together instead!")
        return "Can you explain the basic steps of the algorithm and its time complexity?"

def evaluate_test_answer(question, student_answer, algorithm):
    prompt = f"""
    Evaluate the student's answer to the following test question on {algorithm.name}:

    Question: {question}

    Student's answer: {student_answer}

    Provide a JSON response with the following structure:
    {{
        "passed": true/false,
        "score": 0-100,
        "feedback": "Detailed feedback on the answer, including what was correct and any areas for improvement"
    }}
    """

    try:
        response = model.generate_content(prompt)
        evaluation = json.loads(response.text)
        return evaluation
    except json.JSONDecodeError:
        st.info("I'm having a bit of trouble evaluating your answer precisely. Let's discuss it together!")
        return {
            "passed": False,
            "score": 50,
            "feedback": "Your answer covered some key points, but there's room for improvement. Let's review the algorithm together."
        }
    except Exception as e:
        st.error("Oops! I couldn't evaluate your answer properly. Let's go over the algorithm step by step!")
        return {
            "passed": False,
            "score": 0,
            "feedback": "I apologize, but I couldn't properly evaluate your answer. Let's review the key points of the algorithm together."
        }

def text_to_speech(text):
    tts = gtts.gTTS(text)
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp

def execute_code(code):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code)
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="AI Sorting Algorithm Teacher", page_icon="🧠", layout="wide")
    st.title("🤖 AI Sorting Algorithm Teacher")

    user_models = get_user_models()
    user_id = st.session_state.get("user_id", None)
    
    if user_id is None or user_id not in user_models:
        user_id = str(random.randint(1000, 9999))
        st.session_state.user_id = user_id
        user_models[user_id] = UserModel()

    user_model = user_models[user_id]

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        user_model.set_current_algorithm(random.choice(list(user_model.algorithms.keys())))
        initial_message = f"Hello! 👋 I'm excited to help you learn about sorting algorithms. Let's start with {user_model.current_algorithm.name}. What do you know about this sorting algorithm?"
        st.session_state.messages.append({"role": "assistant", "content": initial_message})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if student_input := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": student_input})
        with st.chat_message("user"):
            st.markdown(student_input)

        assistant_response = get_socratic_response(student_input, user_model)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            audio_file = text_to_speech(assistant_response)
            st.audio(audio_file, format='audio/mp3')

        user_model.add_to_history("Student", student_input)
        user_model.add_to_history("Assistant", assistant_response)

        understood, reasoning = assess_understanding(student_input, assistant_response, user_model)
        if understood:
            st.success(f"Great job! You seem to understand the concept of {user_model.current_concept}.")
            if not user_model.current_concept:
                test_question = generate_test_question(user_model.current_algorithm)
                st.session_state.messages.append({"role": "assistant", "content": f"Fantastic progress! 🎉 Let's test your understanding of {user_model.current_algorithm.name}:\n\n{test_question}"})
                with st.chat_message("assistant"):
                    st.markdown(f"Fantastic progress! 🎉 Let's test your understanding of {user_model.current_algorithm.name}:\n\n{test_question}")
                    audio_file = text_to_speech(f"Fantastic progress! Let's test your understanding of {user_model.current_algorithm.name}. {test_question}")
                    st.audio(audio_file, format='audio/mp3')
        else:
            st.info(f"Let's explore {user_model.current_concept} a bit more. {reasoning}")

    st.sidebar.title("🚀 Learning Progress")
    for alg_name, algorithm in user_model.algorithms.items():
        progress = len(algorithm.understood_concepts) / len(algorithm.concepts)
        st.sidebar.progress(progress, text=f"{alg_name}: {progress:.0%}")

    st.sidebar.title("💻 Code Implementation")
    selected_algorithm = st.sidebar.selectbox("Select an algorithm to implement:", list(user_model.algorithms.keys()))
    
    code = st.sidebar.text_area("Enter your Python code here:", value=user_model.code_implementations.get(selected_algorithm, ""), height=300)
    if st.sidebar.button("Run Code"):
        output = execute_code(code)
        st.sidebar.code(output)
        user_model.add_code_implementation(selected_algorithm, code)

    st.sidebar.title("📚 Study Resources")
    if st.sidebar.button("View Sorting Algorithm Cheat Sheet"):
        st.sidebar.markdown("""
        ### Sorting Algorithm Cheat Sheet
        - **Bubble Sort**: O(n^2), in-place, stable
        - **Insertion Sort**: O(n^2), in-place, stable
        - **Merge Sort**: O(n log n), not in-place, stable
        - **Quick Sort**: O(n log n) average, O(n^2) worst, in-place, not stable
        """)

    if st.sidebar.button("Show Algorithm Visualization"):
        st.sidebar.video("https://www.youtube.com/watch?v=kPRA0W1kECg")

if __name__ == "__main__":
    main()
