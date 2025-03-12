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

### Hugging Face Transformers Text Classification Example
```python
from transformers import pipeline

# Load a pre-trained text classification pipeline
classifier = pipeline("sentiment-analysis")

# Classify a sample text
text = "I love using Hugging Face Transformers!"
result = classifier(text)

print(f"Text: {text}")
print(f"Classification: {result}")
```
This example demonstrates how to use Hugging Face Transformers for text classification using a pre-trained sentiment analysis model.

### Google Cloud AI Speech-to-Text Example
```python
from google.cloud import speech_v1p1beta1 as speech

# Initialize a client
client = speech.SpeechClient()

# Define the audio file to transcribe
audio = speech.RecognitionAudio(uri="gs://my-bucket/audio-file.wav")

# Configure the request
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US"
)

# Perform the transcription
response = client.recognize(config=config, audio=audio)

# Print the transcription
for result in response.results:
    print(f"Transcript: {result.alternatives[0].transcript}")
```
This example demonstrates how to use Google Cloud AI's Speech-to-Text API to transcribe an audio file.

### Deepgram Speech Recognition Example
```python
import deepgram_sdk

# Initialize the Deepgram SDK client
client = deepgram_sdk.Deepgram("your-api-key")

# Define the audio file to transcribe
audio_file_path = "path/to/audio-file.wav"

# Open the audio file
with open(audio_file_path, "rb") as audio_file:
    # Transcribe the audio using Deepgram
    response = client.transcription.sync_prerecorded(audio_file, {
        "punctuate": True,
        "language": "en-US"
    })

# Print the transcription
print(f"Transcript: {response['results']['channels'][0]['alternatives'][0]['transcript']}")
```
This example demonstrates how to use Deepgram for speech recognition and transcription of an audio file.

### LiveKit Real-Time Communication Example
```javascript
import { connect } from 'livekit-client';

// Connect to a LiveKit room
const room = await connect('wss://your-livekit-server', {
  token: 'your-access-token',
});

// Subscribe to room events
room.on('participantConnected', participant => {
  console.log(`${participant.identity} has joined the room.`);
});

room.on('participantDisconnected', participant => {
  console.log(`${participant.identity} has left the room.`);
});

// Publish a local track (e.g., audio)
const localTrack = await navigator.mediaDevices.getUserMedia({ audio: true });
room.localParticipant.publishTrack(localTrack.getAudioTracks()[0]);
```
This example demonstrates how to use LiveKit for real-time communication, including connecting to a room, subscribing to events, and publishing a local audio track.

### Azure DevOps YAML Pipeline Example
```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - script: echo "Building the project..."
    displayName: 'Build Step'

  - script: echo "Running tests..."
    displayName: 'Test Step'

  - script: echo "Deploying the application..."
    displayName: 'Deploy Step'
```
This example demonstrates a simple Azure DevOps pipeline defined in YAML, including steps for building, testing, and deploying an application.
