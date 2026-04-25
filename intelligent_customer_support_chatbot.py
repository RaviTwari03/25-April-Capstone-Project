import os
from fastapi import FastAPI
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chains import RetrievalQA
from dotenv import load_dotenv
import os 

AZURE_ENDPOINT =os.getenv("AZURE_ENPOINT_M3")
API_KEY =os.getenv("API_KEY_M3")
DEPLOYMENT_NAME =  "gpt-4.1"  # IMPORTANT

# -------------------------------
# 🤖 LLM SETUP (AZURE)
# -------------------------------
llm = ChatOpenAI(
    api_key=AZURE_API_KEY,
    base_url=f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}",
    model=DEPLOYMENT_NAME,
    api_version="2024-02-15-preview"
)

# -------------------------------
# 📄 LOAD CSV DATA (RAG)
# -------------------------------
loader = CSVLoader(file_path="customer_support_data.csv")
documents = loader.load()

# -------------------------------
# 🔎 EMBEDDINGS + VECTOR STORE
# -------------------------------
embeddings = OpenAIEmbeddings(
    api_key=AZURE_API_KEY,
    base_url=AZURE_ENDPOINT
)

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -------------------------------
# 🔌 MOCK BACKEND SYSTEMS
# -------------------------------
def get_customer_data(customer_id: str):
    # Simulated CRM response
    return {
        "customer_id": customer_id,
        "name": "Amit",
        "plan": "Premium",
        "last_issue": "Delayed delivery"
    }

def create_ticket(issue: str):
    return f"✅ Ticket created successfully for issue: {issue}"

# -------------------------------
# 🧠 SIMPLE ROUTER LOGIC
# -------------------------------
def route_query(query: str):
    query_lower = query.lower()

    if "ticket" in query_lower or "complaint" in query_lower:
        return create_ticket(query)

    elif "customer" in query_lower or "account" in query_lower:
        data = get_customer_data("C001")
        return f"Customer Info: {data}"

    else:
        # Default → RAG
        return qa_chain.run(query)

# -------------------------------
# 🌐 FASTAPI APP
# -------------------------------
app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot Running 🚀"}

@app.post("/chat")
def chat(req: ChatRequest):
    response = route_query(req.query)
    return {"response": response}