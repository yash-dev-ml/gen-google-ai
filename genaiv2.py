import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import random
import uuid  # For generating unique user IDs
import gtts
from io import BytesIO
import io
import contextlib

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

class StudentModel:
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

@st.cache_resource
def get_student_model():
    return StudentModel()

def generate_user_id():
    return str(uuid.uuid4())

if 'user_id' not in st.session_state:
    st.session_state.user_id = generate_user_id()

def get_user_model(user_id):
    if user_id not in st.session_state:
        st.session_state[user_id] = get_student_model()
    return st.session_state[user_id]

def get_socratic_response(student_input, student_model):
    conversation = "\n".join([f"{entry['role']}: {entry['content']}" for entry in student_model.conversation_history])

    prompt = f"""
    You are an AI teaching assistant specializing in sorting algorithms. Your goal is to guide the student to understand and master sorting algorithms through Socratic questioning and clear explanations when necessary.

    Current algorithm: {student_model.current_algorithm.name}
    Current concept: {student_model.current_concept}
    Understood concepts: {', '.join(student_model.current_algorithm.understood_concepts)}

    Recent conversation:
    {conversation}

    Student input: {student_input}

    Based on the student's progress and the conversation history:
    1. If introducing a new concept, provide a brief explanation followed by a thought-provoking question.
    2. If the student shows understanding, ask a more challenging question or present a scenario to apply the concept.
    3. If the student seems confused, break down the concept and ask simpler questions to build understanding.
    4. Regularly check for understanding and provide encouragement.
    5. If all concepts for the current algorithm are understood, prepare a short test question to confirm mastery.

    Respond in the voice of a supportive teacher, providing guidance and asking questions to promote critical thinking.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error("Oops! I'm having a bit of trouble thinking right now. Let's try a different approach!")
        return "I apologize for the confusion. Could you tell me what specific aspect of the algorithm you'd like to explore?"

def assess_understanding(student_input, assistant_response, student_model):
    prompt = f"""
    Analyze the following student response and determine if it demonstrates understanding of the current concept ({student_model.current_concept}) for the algorithm ({student_model.current_algorithm.name}).

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
            student_model.update_understanding(student_model.current_concept)
        
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
            "feedback": "I apologize, but I couldn't properly evaluate your answer. Let's review the key concepts."
        }

def text_to_speech(text):
    tts = gtts.gTTS(text, lang='en')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

def main():
    st.title("Sorting Algorithm Tutor")
    
    user_id = st.session_state.user_id
    student_model = get_user_model(user_id)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        student_model.set_current_algorithm(random.choice(list(student_model.algorithms.keys())))
        initial_message = f"Hello! 👋 I'm excited to help you learn about sorting algorithms. Let's start with {student_model.current_algorithm.name}. What do you know about this sorting algorithm?"
        st.session_state.messages.append({"role": "assistant", "content": initial_message})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if student_input := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": student_input})

        # Generate Socratic response based on student input
        assistant_response = get_socratic_response(student_input, student_model)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Assess understanding based on the student's input and assistant's response
        understood, reasoning = assess_understanding(student_input, assistant_response, student_model)
        if understood:
            st.session_state.messages.append({"role": "assistant", "content": "Great! You've understood the concept. Let's move on to the next topic or try a test question."})
            student_model.update_understanding(student_model.current_concept)
            student_model.current_concept = next(iter(set(student_model.current_algorithm.concepts) - student_model.current_algorithm.understood_concepts), None)

            if not student_model.current_concept:
                next_algorithm = student_model.get_next_algorithm()
                if next_algorithm:
                    student_model.set_current_algorithm(next_algorithm)
                    st.session_state.messages.append({"role": "assistant", "content": f"Awesome job! Now, let's start learning about {student_model.current_algorithm.name}."})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "You've covered all the concepts! Would you like to take a test or review what we've learned?"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"I appreciate your effort. Let's revisit the concept of {student_model.current_concept}."})
        
        # Provide a test question if all concepts are understood
        if not student_model.current_concept:
            question = generate_test_question(student_model.current_algorithm)
            st.session_state.messages.append({"role": "assistant", "content": f"Here is a test question to review your understanding:\n\n{question}"})

        # Evaluate student's test answer
        if test_answer := st.text_input("Your answer to the test question"):
            evaluation = evaluate_test_answer(question, test_answer, student_model.current_algorithm)
            st.session_state.messages.append({"role": "assistant", "content": f"Evaluation:\n\nScore: {evaluation['score']}%\n\nFeedback: {evaluation['feedback']}"})
            
            if evaluation["passed"]:
                st.session_state.messages.append({"role": "assistant", "content": "Well done! You've passed the test. Would you like to learn about another algorithm or review something else?"})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Let's go over the concepts again to help you improve."})

        # Play audio for assistant's responses
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                audio_file = text_to_speech(message["content"])
                st.audio(audio_file, format='audio/mp3')

if __name__ == "__main__":
    main()
