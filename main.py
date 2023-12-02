import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import statement
from langchain.embedding.openai import OpenAIEmbeddings  # Corrected import statement
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# (The previously provided 'read_and_textify' function remains unchanged)

st.set_page_config(layout="centered", page_title="doc_reader")
st.header("Doc Reader")
st.write("---")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])

if uploaded_files is None:
    st.info("Upload files to analyze")
elif uploaded_files:
    st.write(f"{len(uploaded_files)} document(s) loaded..")

textify_output = read_and_textify(uploaded_files)
documents = textify_output[0]
sources = textify_output[1]

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openad-api-key"])  # Corrected class name and parameter
vStore = Chroma.from_text(documents, embeddings, metadatas=[{"source": s} for s in sources])

model_name = "gpt-3.5-turbo"
retriever = vStore.as_retriever()
retriever.search_kwargs = {'k': 2}

llm = OpenAI(model_name=model_name, openai_api_key=st.secrets["open_api_key"])  # Corrected parameter name and removed 'streaming=True'
model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.header('Ask your data')
user_q = st.text_area("Enter your questions here")  # Corrected quotation mark

if st.button("Get Response"):
    try:
        with st.spinner('Model is working on it...'):
            result = model({"question": user_q}, return_only_outputs=True)  # Corrected assignment operator
            st.subheader('Your response:')
            st.write(result['answer'])
            st.subheader('Source pages:')
            st.write(result['sources'])  # Corrected variable name
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error('Oops, the GPT response resulted in an error : (Please try again.')
