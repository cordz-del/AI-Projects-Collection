# AI-Projects-Collection
A collection of AI projects demonstrating various skills and frameworks.

## Introduction

This repository contains a collection of AI projects that demonstrate various skills and frameworks. Each project showcases the practical application of different AI technologies, including LLM frameworks, RAG frameworks, AI automation, and more.

## Skills & Certifications

This repository demonstrates the following skills and certifications:

- **LLM Frameworks**: OpenAI GPT (GPT-3, GPT-4, ChatGPT), Anthropic Claude, Meta's LLaMA, Google PaLM/Gemini, Hugging Face Transformers, Cohere AI.
- **RAG Frameworks**: LangChain, LlamaIndex, Haystack, Chroma, Pinecone, Weaviate, FAISS.
- **AI Automation & CI/CD**: Azure DevOps, GitHub Actions, Jenkins, Terraform, AWS SageMaker, Google Vertex AI, Docker & Kubernetes.
- **Additional Tools & APIs**: FastAPI/Flask, AWS AI Services, Google Cloud AI APIs, Deepgram, LiveKit.

### Certifications:
- Stanford Machine Learning Certification
- Google Cloud Generative AI Certification
- Duke University LLMOps Certification
- API Testing Foundations Certification
- Test Automation Certification

## Code Snippet Examples

### OpenAI GPT Text Generation Example
```python
import openai

# Set up the OpenAI API key
openai.api_key = "your-api-key"

# Define a prompt for text generation
prompt = "Once upon a time in a land far, far away"

# Generate text using GPT-3
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=100
)

# Print the generated text
print(response.choices[0].text.strip())
```
This example demonstrates how to use OpenAI GPT-3 for text generation by providing a prompt and generating a continuation.

### LangChain RAG Example
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize the vector store with Chroma
vector_store = Chroma("my_collection", embeddings=OpenAIEmbeddings())

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_vector_store(vector_store)

# Ask a question and get an answer
question = "What is the capital of France?"
answer = qa_chain.run(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
```
This example demonstrates how to use LangChain for retrieval-augmented generation, integrating a vector store with Chroma and OpenAI embeddings.

### AWS SageMaker Model Deployment Example
```python
import boto3
from sagemaker import Session
from sagemaker.tensorflow import TensorFlowModel

# Initialize a SageMaker session
sagemaker_session = Session()

# Define the S3 path to the model artifact
model_artifact = "s3://my-bucket/my-model.tar.gz"

# Create a TensorFlow model
model = TensorFlowModel(
    model_data=model_artifact,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    framework_version="2.3",
    sagemaker_session=sagemaker_session
)

# Deploy the model to a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("Model deployed successfully.")
```
This example demonstrates how to deploy a TensorFlow model to an AWS SageMaker endpoint.

### FastAPI Example
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "value": "This is item " + str(item_id)}

# To run the app, use the command: uvicorn main:app --reload
```
This example demonstrates how to create a simple API with FastAPI, including a root endpoint and a parameterized endpoint.

### Dockerfile Example for Python Application
```Dockerfile
# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the required Python packages
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "app.py"]
```
This Dockerfile sets up a Python environment, installs dependencies, and runs a Python application.
