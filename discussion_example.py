from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

model_name = os.environ.get('MODEL_NAME')
system_prompt_directory = os.environ.get('SYSTEM_PROMPT_DIRECTORY')

llm = Builder() \
.with_llm(Settings.OLLAMA, model_name=model_name, system_prompt_file=system_prompt_directory) \
.build_llm()

def chat():
    query = input(">>> ")
    if query == "quit" or query == "bye" : 
        print('🤖 : See you soon 👋')
        return
    response = llm.generate({"question": query})
    print('🤖 : ', response)
    return chat()

chat()