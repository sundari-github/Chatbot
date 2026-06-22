from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# Read API key from param.env file. It has the value in key=value format
# Key is OPEN_API_KEY
with open("param.env", "r") as f:
    for line in f:
        if line.startswith("OPEN_API_KEY="):
            OPEN_API_KEY = line.split("=")[1].strip()
            break

# Check if the API key is loaded
if not OPEN_API_KEY:
    st.error("API key not found. Please check your param.env file.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.set_page_config(page_title= "PDF Chatbot", page_icon= "📄")
st.sidebar.title("PDF Chatbot")
file = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])

st.title("PDF Chatbot")

@st.cache_resource
def process_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=1000, 
        chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create embeddings and store in FAISS
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPEN_API_KEY
    )
    return FAISS.from_texts(chunks, embeddings)

# Extract contents of the file and chunk it
if file is not None:
    vector_store = process_pdf(file)
      
    st.sidebar.success("PDF processed successfully!") 

    # Read the user input from the text box
    user_question = st.text_input("Ask a question about your PDF:")

    # Embed the user's question
    # The below is not required. The embeddings happen automatically under the hood inside the 
    # Retriever/VectorStore classes when you invoke the chain
    # user_embedding = embeddings.embed_query(user_question)

    # Perform similarity search. The below method is typically used 
    # where you just want to get documents back as a list and don't need to pass them into a larger pipeline.
    # docs = vector_store.similarity_search(user_embedding)

    # A Retriever is a broader abstraction (interface) designed specifically to be plugged into 
    # LangChain's chains and pipelines.
    # MMR (Maximal Marginal Relevance) is a search algorithm that finds documents that are 
    # both relevant to the query and dissimilar to each other.
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}, # Get the top 5 chunks for better context
        search_type="mmr", # Perform Maximal Marginal Relevance search
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant who will answer questions based on the PDF\n"
                        "Guidelines\n"
                            "1. Make the answer precise and avoid irrelevant details\n"
                            "2. Include relevant details from the document if required\n"
                            "3. Use only the information from the provided context\n"
                            "4. Do not use any outside knowledge\n"
                            "Context : {context}\n"),
            ("human", "{question}")
        ]
    )


    llm = ChatOpenAI(
        model_name="gpt-4o", # Changed from gpt-5 to a valid model
        openai_api_key=OPEN_API_KEY,
        temperature=0, # For deterministic output
        max_tokens=1000, # For limiting the output length
    )

    # Defines a chain with 4 components: input, prompt, LLM, and output parser
    # question is the user input, RunnablePassthrough() is a utility that passes the input to the next component (context and question)
    # The retriever is passed through the format_docs function to format the documents
    chain = (
        {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        with st.spinner("Thinking..."):
            response = chain.invoke(user_question)
            st.write(response)
    else:
        st.info("Ask a question about your PDF") 
        
else:
    st.sidebar.warning("Please upload a PDF file.")




