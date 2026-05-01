import random
import os

intents = {
    "greetings": {
        "inputs": ["hello", "hi", "hey", "salam", "assalamualaikum"],
        "responses": [
            "Hello! How can I assist you today?",
            "Hi there! What can I do for you?",
            "Hey! How may I help?",
            "Assalamualaikum! Kya madad chahiye?",
            "Hello! I'm here to help you.",
            "Hi! Aap kaise hain?",
            "Welcome! Bataiye kya chahiye?",
            "Hey there! Need any help?",
            "Salam! Main madad ke liye hoon.",
            "Hello! Let's get started."
        ]
    },
    "thanks": {
        "inputs": ["thank", "thanks", "thankyou", "shukriya"],
        "responses": [
            "You're welcome!",
            "No problem at all!",
            "Glad I could help!",
            "Khushi hui madad karke!",
            "Anytime!",
            "Zaroor!",
            "It's my pleasure!",
            "Happy to assist!",
            "You're most welcome!",
            "No worries!"
        ]
    },
    "help": {
        "inputs": ["help", "support", "problem"],
        "responses": [
            "Sure, please tell me your problem.",
            "I'm here to help. What do you need?",
            "Aap apni problem bata sakte hain.",
            "Let me know how I can assist.",
            "Tell me the issue, I'll try to help.",
            "Main madad ke liye hoon, batayein.",
            "What kind of help do you need?",
            "Feel free to explain your issue.",
            "Yes, I'm listening.",
            "Go ahead, I'm here."
        ]
    },
    "yes": {
        "inputs": ["yes", "haan"],
        "responses": [
            "Great!",
            "Perfect!",
            "Awesome!",
            "Zabardast!",
            "Good to hear!",
            "Nice!",
            "Excellent!",
            "Alright!",
            "That's good!",
            "Okay, moving forward."
        ]
    },
    "no": {
        "inputs": ["no", "nahi"],
        "responses": [
            "Okay, no problem.",
            "Alright, let me know if needed.",
            "No worries.",
            "Theek hai.",
            "Got it.",
            "Alright.",
            "Understood.",
            "Fine.",
            "Okay then.",
            "As you wish."
        ]
    },
    "bye": {
        "inputs": ["bye", "goodbye", "allah hafiz"],
        "responses": [
            "Goodbye! Take care.",
            "See you later!",
            "Allah Hafiz!",
            "Bye! Have a great day.",
            "Take care!",
            "Catch you later!",
            "Khuda Hafiz!",
            "Stay safe!",
            "Bye! Come back anytime.",
            "See you soon!"
        ]
    },
    "one": {
        "inputs": ["one", "1"],
        "responses": [
            "You showed number One!",
            "I see number 1!",
            "That's number One!"
        ]
    },
    "two": {
        "inputs": ["two", "2"],
        "responses": [
            "You showed number Two!",
            "I see number 2!",
            "That's number Two!"
        ]
    },
    "three": {
        "inputs": ["three", "3"],
        "responses": [
            "You showed number Three!",
            "I see number 3!",
            "That's number Three!"
        ]
    },
    "four": {
        "inputs": ["four", "4"],
        "responses": [
            "You showed number Four!",
            "I see number 4!",
            "That's number Four!"
        ]
    },
    "five": {
        "inputs": ["five", "5"],
        "responses": [
            "You showed number Five!",
            "I see number 5!",
            "That's number Five!"
        ]
    }
}

def rule_based_response(text):
    text = text.lower()
    for intent in intents.values():
        for word in intent["inputs"]:
            if word in text:
                return random.choice(intent["responses"])
    return None

def ai_response(text):
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        You are a helpful assistant for a sign language system.
        Reply in short, simple, and friendly sentences.
        User: {text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "I understood your sign! How can I help you further?"

def chatbot_response(text):
    response = rule_based_response(text)
    if response is None:
        return ai_response(text)
    return response
