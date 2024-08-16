import speech_recognition as sr
import pyttsx3
from langserve import RemoteRunnable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 기존 코드 유지
llm = RemoteRunnable("https://worthy-huge-mustang.ngrok-free.app/llm/")

# FAISS 벡터 저장소 생성 함수 (PDF 파일 처리)
def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

class RunnablePassthrough:
    def __call__(self, input_data):
        return input_data

# RAG 체인 생성 함수
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'fetch_k': 10})
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | PromptTemplate(input_variables=["context", "question"], template="{context}\n\n{question}")
        | llm
    )
    return rag_chain

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
    # FAISS 벡터 저장소 생성
    pdf_path = "ComputerSoftware.pdf"  # PDF 문서 디렉토리 경로 설정
    vector_store = create_vector_store(pdf_path)

    # RAG 체인 생성
    rag_chain = create_rag_chain(vector_store)

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        print("Listening...")
        result = recognize_speech_from_mic(recognizer, mic)
        if result["transcription"]:
            print("You said: {}".format(result["transcription"]))
            try:
                # RAG 체인을 사용하여 응답 생성
                rag_response = rag_chain.invoke(result["transcription"])
                print("RAG response: {}".format(rag_response))
                speak_text(rag_response)
            except Exception as e:
                print("Error during RAG chain execution: {}".format(e))
        if not result["success"]:
            print("I didn't catch that. What did you say?\n")
        if result["error"]:
            print("ERROR: {}".format(result["error"]))
            break