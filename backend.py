import os
import google.generativeai as genai

class F1KnowledgeBase:
    def __init__(self, user_login, api_key=""):
        self.user_login = user_login
        self.API_KEY = api_key or os.getenv("GEMINI_API_KEY")
        if self.API_KEY:
            genai.configure(api_key=self.API_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        )

    def generate_response(self, prompt):
        response = self.model.generate_content([
            "You are a Formula 1 chatbot, so answer accordingly",
            f"input: {prompt}",
            "output: ",
        ])
        return response.text

# Example usage
if __name__ == "__main__":
    kb = F1KnowledgeBase(user_login="example_user")
    print(kb.generate_response("Who are you?"))