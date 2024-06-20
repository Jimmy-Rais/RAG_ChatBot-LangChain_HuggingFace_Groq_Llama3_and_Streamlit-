#LangchainBot
import streamlit as st # type: ignore
from PyPDF2 import PdfReader
import os
#Langchain libraries(Loaders,splitters,embeddings,prompt,llm,retrievalqa,vectorstores)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("groq_api_key") 
#Extract text from pdf
def extract_pdf(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf) #List of pages
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
#Split the text in smaller chunks using Langchain RecursiveTextSplitters
def get_split_text(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )
    chunks=splitter.split_text(text)
    return chunks
#Convert the chunks into Embeddings using HuggingFaceInst ructEmbeddings
#Store the embeddings into a vectorstore,Chroma db in this case
def get_vectorstore(chunks):
    embedding=HuggingFaceInstructEmbeddings(
         model_name="hkunlp/instructor-xl",
         model_kwargs={"device": "cuda"}
    )
    vectordb=Chroma.from_documents(documents=chunks,embedding=embedding)
    vectordb.save_local("chroma_index")
    return vectordb
#Conversational chain
def get_conversational_chain():
    #Prompt template, {content} is a place holder for the documents and {question} a place holder for the question
    template= """
    Answer the question based on the provided ressources only.
    Give the most relevant answer.
     {context}
     Question: {question}
     Answer:
    """
    qa_prompt=ChatPromptTemplate.from_template(template)
    #Llm initialization
    llm=ChatGroq(
    model_name="Llama3-8b-8192",
    groq_api_key=API_KEY,
    temperature=0)
    #retriever=vectordb.as_retriever()
    #Qa_chain initialization
    qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    #retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":qa_prompt}
    )
    return qa_chain
def get_user_input(question):  
    #Embeddings
    embeddings=HuggingFaceInstructEmbeddings(
         model_name="hkunlp/instructor-xl",
    )
    #Load the saved vectorstore
    new_db = Chroma.load_local("chroma_index", embeddings)
    #Perform a similarity search between the user question and the documents stored in the vectorstore
    docs = new_db.similarity_search(question)
    chain=get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])
#User inteface
def main():
    st.set_page_config("Chat PDF")
    st.header("Let's crack that exam together💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        get_user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text =extract_pdf(pdf_docs)
                text_chunks = get_split_text(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()