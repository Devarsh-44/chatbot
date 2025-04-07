import os
import google.generativeai as genai

class F1KnowledgeBase:
    def __init__(self, user_login, api_key=""):
        self.user_login = user_login
        self.API_KEY = api_key or os.getenv("API_KEY")  # Changed to match .env variable name
        if not self.API_KEY:
            raise ValueError("API_KEY environment variable is not set")
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

    def text_to_image(self, text, image_path):
        # Placeholder: Text-to-image generation is not supported by Gemini API
        # You can integrate a different API (e.g., DALL-E, Stable Diffusion) here
        raise NotImplementedError("Text-to-image generation is not implemented in this version")

# Example usage
if __name__ == "__main__":
    kb = F1KnowledgeBase(user_login="example_user")
    print(kb.generate_response("Who are you?"))