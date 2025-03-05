import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import time
import nltk
import io
import base64
import numpy as np
import pandas as pd
import requests
from llama_index.core import ServiceContext, SimpleDirectoryReader, GPTVectorStoreIndex, PromptHelper, VectorStoreIndex, KeywordTableIndex, StorageContext, load_index_from_storage
from llama_index.llms.vllm import Vllm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from transformers import AutoTokenizer
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from faster_whisper import WhisperModel
import wave
from piper.voice import PiperVoice

class InferlessPythonModel:
    def initialize(self):
        self.audio_file = "output.mp3"
        # Initialize tokenizer
        model_id = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Initialize LLM
        self.llm = Vllm(
            model="NousResearch/Hermes-2-Pro-Llama-3-8B",
            max_new_tokens=256,
            top_k=10,
            top_p=0.95,
            temperature=0.1,
            vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.9},
        )

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024

        #Load the Dataset to Pinecone
        self.load_dataset()

        # Initialize Pinecone
        self.pc = Pinecone(api_key="YOUR-API-KEY")
        self.index = self.pc.Index("document")

        # Initialize vector store and query engine
        self.vector_store = PineconeVectorStore(pinecone_index=self.index)
        self.index = GPTVectorStoreIndex.from_vector_store(self.vector_store)
        self.query_engine = self.index.as_query_engine()

        # Initialize Whisper model
        self.model_size = "large-v3"
        self.model_whisper = WhisperModel(self.model_size, device="cuda", compute_type="float16")

        # Ensure the onnx_models directory exists
        self.model_dir = "onnx_models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Download the Piper voice model if it doesn't exist
        self.model_path = os.path.join(self.model_dir, "en_US-lessac-medium.onnx")
        self.model_json_path = os.path.join(self.model_dir, "en_US-lessac-medium.onnx.json")
        self.download_model()

        # Initialize Piper voice model
        self.voice = PiperVoice.load(self.model_path, use_cuda=True)
    
    def base64_to_mp3(self, base64_data, output_file_path):
        # Convert base64 audio data to mp3 file
        mp3_data = base64.b64decode(base64_data)
        with open(output_file_path, "wb") as mp3_file:
            mp3_file.write(mp3_data)

    def load_dataset(self):
        # Setup dataset directory and file path
        dataset_dir = "dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, "SpotifyCustomerSupport.txt")
        
        # Download dataset if it doesn't exist
        if not os.path.exists(dataset_path):
            url = "https://github.com/inferless/Customer-Service-Voicebot/raw/main/SpotifyCustomerSupport.txt"
            response = requests.get(url)
            response.raise_for_status()
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
        
            # Load documents and initialize Pinecone
            documents = SimpleDirectoryReader(dataset_dir).load_data()
            pc = Pinecone(api_key="153e3e06-a636-4925-bd3f-82b3349d59eb")
            pc.create_index(
                name="document",
                dimension=384,
                metric="euclidean",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            index = pc.Index("document")
            
            # Setup vector store and storage context
            vector_store = PineconeVectorStore(pinecone_index=index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def download_model(self):
        if not os.path.exists(self.model_path):
            url_list = ["https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
            , "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"]
            download_items = [self.model_path,self.model_json_path]
            for idx in range(len(url_list)):
                response = requests.get(url_list[idx])
                response.raise_for_status()  # Check if the request was successful
                with open(download_items[idx], 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {download_items[idx]}")

    def infer(self, inputs):
        audio_data = inputs["audio_base64"]
        
        #Convert the audio from base64 to .mp3
        self.base64_to_mp3(audio_data, self.audio_file)

        # Transcribe audio file
        segments, info = self.model_whisper.transcribe(self.audio_file, beam_size=5)
        user_text = ''.join([segment.text for segment in segments])

        # Prepare messages for chat template
        messages = [
            {"role": "system", "content": "You are Customer Support Assistant."},
            {"role": "user", "content": user_text}
        ]

        # Generate input for the LLM
        gen_input = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)

        # Query the vector store
        response = self.query_engine.query(gen_input)

        # Synthesize response to audio
        byte_stream = io.BytesIO()
        with wave.open(byte_stream, "wb") as wav_file:
            # Set WAV file parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(22050)  # Sample rate

            # Synthesize the speech and write to byte stream
            self.voice.synthesize(response.response, wav_file)

        # Get the byte stream's content
        audio_bytes = byte_stream.getvalue()

        # Encode audio bytes to Base64 string
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {"generated_audio_base64": audio_base64,
               "question":user_text,
               "answer":response.response}

    def finalize(self):
        # Clear GPU memory (implementation depends on the framework used)
        pass
