import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from urllib.request import urlopen
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# device = "cpu"
def get_pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        # set chunk size to 1000 characters
        chunk_size=1000,
        # makes sure you don't lose the meaning of a sentence to cut chunks in the middle of a sentence
        chunk_overlap=200,
        # using default length function in python
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # use Azure version
    # llm = AzureChatOpenAI()
    # if wanna use a llm model from huggingface
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length": 512})
    # chatbot has a memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # initialize the session
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_prompt(user_prompt):
    response = st.session_state.conversation({'question': user_prompt})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

def get_html_xml_text(url_or_file):
    text = ""
    # get text from url capability
    if url_or_file.startswith("http"):
        url_or_file = urlopen(url).read()
    soup = BeautifulSoup(url_or_file, "html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Trax Chatbot", page_icon=":books:", layout="wide")
    # initialize conversation state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.header("Trax Chatbot")
    user_prompt = st.text_input("Search Building Codes")
    if user_prompt:
        handle_user_prompt(user_prompt)

    with st.sidebar:
        st.subheader("Building Code Docs")
        docs = st.file_uploader("upload PDFs, HTMLs, or XMLs", accept_multiple_files=True)
        if st.button("Process Docs"):
            with st.spinner("processing docs..."):

                # get PDF, HTML, or XML text
                if docs:
                    pdf_docs = [doc for doc in docs if doc.name.endswith(".pdf")]
                    html_xml_docs = [doc for doc in docs if doc.name.endswith(".html") or doc.name.endswith(".xml")]
                    if pdf_docs:
                        raw_text = get_pdf_text(pdf_docs)
                    elif html_xml_docs:
                        raw_text = get_html_xml_text(html_xml_docs[0])
                    else:
                        st.error("Please upload PDF, HTML, or XML files")
                        return
                    # st.write(raw_text)
                # get text chunks from PDF, HTML, or XML
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                # conversation = get_conversation_chain(vectorstore)

                # if you don't wanna initialize all parameters at the start of streamlit app, do the following
                st.session_state.conversation = get_conversation_chain(vectorstore)

    # can access session state anywhere in the app
    # make variables persistent across sessions
    # st.session_state.conversation
if __name__ == '__main__':
    main()