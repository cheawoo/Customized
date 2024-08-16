import speech_recognition as sr
from langchain_community.llms import Ollama
import pyttsx3

model = Ollama(model="llama3:latest")

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`."""
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `speech_recognition.Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `speech_recognition.Microphone` instance")

    # Adjust the recognizer sensitivity to ambient noise and record audio from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio, language='ko-KR')
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

def speak_text(text):
    """Convert text to speech and play it through speakers."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        print("Listening...")
        result = recognize_speech_from_mic(recognizer, mic)
        if result["transcription"]:
            print("You said: {}".format(result["transcription"]))
            Ollama_respone = model.invoke(result["transcription"])
            speak_text(Ollama_respone)
        if not result["success"]:
            print("I didn't catch that. What did you say?\n")
        if result["error"]:
            print("ERROR: {}".format(result["error"]))
            break
