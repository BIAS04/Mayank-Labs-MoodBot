from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="mistral-small-latest", 
    model_provider="mistralai",
    temperature=0
)

# print(model)

response = model.invoke("nethwort of elon musk ?")


print(response.content)
