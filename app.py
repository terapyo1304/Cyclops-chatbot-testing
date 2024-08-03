from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from io import StringIO
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
from langchain.chains.question_answering import load_qa_chain
from llama_cpp import Llama

#set variables
PINECONE_API_KEY='1470404d-c8d1-47d5-a682-539414237e46'
PINECONE_API_ENV ='us-east-1'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"


# Streamlit app layout
st.title("Cyclops Chatbot")
pdf_url = st.text_input("Enter the URL of a PDF file:")

#Load data for analysis
loader = PyPDFLoader(pdf_url)
data = loader.load()

#split data into chunks of 500 tokens
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs=text_splitter.split_documents(data)

#Define embeddings model
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# initialize pinecone
pinecone_instance=pinecone.Pinecone(
   api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchainpinecone" # put in the name of your pinecone index here

docsearch=Pinecone.from_texts([t.page_content for t in docs], embedding=embeddings, index_name=index_name)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

#Download Llama 2 model
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

#define Llama 2 model
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=False,
)

chain=load_qa_chain(llm, chain_type="stuff")

#Question to be entered by user
query = st.text_input("Enter your question")
if st.button("Get Response"):
    docs=docsearch.similarity_search(query)
    response=chain.run(input_documents=docs, question=query)
    st.write(response)







