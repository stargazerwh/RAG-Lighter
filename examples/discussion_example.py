from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from dotenv import load_dotenv
import os

load_dotenv()
Settings.setup_logging()

model_name = os.environ.get('MODEL_NAME')
system_prompt_directory = os.environ.get('SYSTEM_PROMPT_DIRECTORY')

llmOllama = Builder() \
.with_llm(Settings.OLLAMA, model_name=model_name, system_prompt_file=system_prompt_directory) \
.build_llm()

# llmLMStudio = Builder() \
# .with_llm(Settings.LMStudio, model_name=model_name, system_prompt_file=system_prompt_directory) \
# .build_llm()

def chat():
    query = input(">>> ")
    if query == "quit" or query == "bye" : 
        print('🤖 : See you soon 👋')
        return
    response = llmOllama.generate({"question": query}, stream=False)
    # response = llmLMStudio.generate({"question": query})
    print('🤖 : ', response)
    return chat()

def chat_streaming():
    query = input(">>> ")
    if query == "quit" or query == "bye" : 
        print('See you soon 👋')
        return
    stream = llmOllama.generate({"question": query}, stream=True)
    for chunk in stream:                                                                                                                                                   
      print(chunk, end='', flush=True) 
    return chat()

chat()
