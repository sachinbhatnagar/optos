import os
from dotenv import load_dotenv
from picamera2 import Picamera2
import json
import logging
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Configuration
RESOLUTION = (640, 480)
API_ENDPOINT = "https://api.google.com/gemini-flash"  # Replace with actual API endpoint
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
USER_PROMPT = "Describe what you see in this image."

# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You are a pair of eyes for a blind person. You will be provided with an image. Your job is to describe what you see in 2 - 3 lines and also advise the person if there are any perceived threats. Act in the first person and like a friend.",
)

chat_session = model.start_chat(history=[])

files = []


def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def capture_photo():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration(main={"size": RESOLUTION}))
        picam2.start()
        picam2.capture_file("image.jpg")
        picam2.stop()
        logging.info("Photo captured successfully")
        return True
    except Exception as e:
        logging.error(f"Error capturing photo: {str(e)}")
        return False


def send_to_api():
    try:
        with open("image.jpg", "rb") as image_file:
            files = [
                upload_to_gemini("image.jpg", mime_type="image/jpeg"),
            ]

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            files[0],
                        ],
                    }
                ]
            )

            response = chat_session.send_message(
                "Describe what you see in this image in the first person."
            )

            if response.status_code == 200:
                return response.text
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        logging.error(f"Error sending to API: {str(e)}")
        return None


def main():
    if capture_photo():
        api_response = send_to_api()
        if api_response:
            print("API Response:")
            print(api_response)
        else:
            print("Failed to get API response")
    else:
        print("Failed to capture photo")


if __name__ == "__main__":
    main()
