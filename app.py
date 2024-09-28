import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the conversation chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say, "The answer is not available in the context." Do not provide the wrong answer.

    Context:\n{context}
    Question:\n{question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process the user's input and generate answers
def user_input(user_question, Question, Answer):
    # Append the new question to the Question list
    Question.append(user_question)

    # Load the saved FAISS index and perform similarity search
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain and generate a response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Append the response to the Answer list
    Answer.append(response["output_text"])

    return Question, Answer

# Main function to handle the Streamlit UI
def main():
    # Initialize session state to store question-answer history
    if "Question" not in st.session_state:
        st.session_state.Question = []
    if "Answer" not in st.session_state:
        st.session_state.Answer = []

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Sidebar to upload PDFs and process them
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Create a form for the user question input
    with st.form(key='question_form', clear_on_submit=True):
        user_question = st.text_input("Ask a Question from the PDF Files", key='user_question')
        submit_button = st.form_submit_button(label='Submit')

    # If the user clicks the submit button
    if submit_button and user_question:
        # Process the user's input and update the session state
        st.session_state.Question, st.session_state.Answer = user_input(user_question, st.session_state.Question, st.session_state.Answer)

    # Display all the question-answer history
    if st.session_state.Question and st.session_state.Answer:
        for i in range(len(st.session_state.Question)):
            st.write(f"**Question {i+1}:** {st.session_state.Question[i]}")
            st.write(f"**Answer {i+1}:** {st.session_state.Answer[i]}")

if __name__ == "__main__":
    main()
