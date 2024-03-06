# imports
import streamlit as st
import os, tempfile
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.agents import create_pandas_dataframe_agent
import asyncio

st.set_page_config(page_title="CSV AI", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def home_page():
    st.write("""Select any one feature from above sliderbox: \n
    1. Chat with CSV \n
    2. Summarize CSV \n
    3. Analyze CSV  """)

@st.cache_resource()
def retriever_func(uploaded_file):
    if uploaded_file :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200, 
                        add_start_index=True
                        )
        all_splits = text_splitter.split_documents(data)

        
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    if not uploaded_file:
        st.info("Please upload CSV documents to continue.")
        st.stop()
    return retriever, vectorstore

def chat(temperature, model_name):
    st.write("# Talk to CSV")
    # Add functionality for Page 1
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    retriever, vectorstore = retriever_func(uploaded_file)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming=True)
        
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    store = {}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Use the following pieces of context to answer the question at the end.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    runnable = prompt | llm
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    async def chat_message():
        if prompt := st.chat_input():
            if not user_api_key: 
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            contextt = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in contextt)
            #msg = 
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                text_chunk = ""
                async for chunk in with_message_history.astream(
                        {"context": context, "input": prompt},
                        config={"configurable": {"session_id": "abc123"}},
                    ):
                    text_chunk += chunk.content
                    message_placeholder.markdown(text_chunk)
                    #st.chat_message("assistant").write(text_chunk)
                st.session_state.messages.append({"role": "assistant", "content": text_chunk})
        if reset:
            st.session_state["messages"] = []
    asyncio.run(chat_message())


def summary(model_name, temperature, top_p):
    st.write("# Summary of CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        # encoding = cp1252
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap=100)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)

        os.remove(tmp_file_path)
        gen_sum = st.button("Generate Summary")
        if gen_sum:
            # Initialize the OpenAI module, load and run the summarize chain
            llm = ChatOpenAI(model_name=model_name, temperature=temperature)
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",

                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )
            result = chain({"input_documents": texts}, return_only_outputs=True)

            st.success(result["output_text"])


def analyze(temperature, model_name):
    st.write("# Analyze CSV")
    #st.write("This is Page 3")
    # Add functionality for Page 3
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    #.write(uploaded_file.name)
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        df = pd.read_csv(tmp_file_path)
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        agent = create_pandas_dataframe_agent(llm, df, agent_type="openai-tools", verbose=True)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="What are the names of the columns?"):
            if not user_api_key: 
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            msg = agent.invoke({"input": prompt, "chat_history": st.session_state.messages})
            st.session_state.messages.append({"role": "assistant", "content": msg["output"]})
            st.chat_message("assistant").write(msg["output"])
        if reset:
            st.session_state["messages"] = []


# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 CSV AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting, Analyzing and Summarizing CSV Files!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    global user_api_key
    # #
    # st.sidebar.write("---Made with ❤️---")
    # st.sidebar.write("---")
    if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
        user_api_key = os.environ["OPENAI_API_KEY"]
        st.success("API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password", key="openai_api_key"
        )
        if user_api_key:
            st.sidebar.success("API key loaded", icon="🚀")

    os.environ["OPENAI_API_KEY"] = user_api_key

    

    # Execute the home page function
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k","gpt-3.5-turbo-16k","gpt-4-1106-preview"]
    max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.9
    TEMPERATURE_STEP = 0.01
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    # freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
    temperature = st.sidebar.slider(
                label="Temperature",
                min_value=TEMPERATURE_MIN_VALUE,
                max_value=TEMPERATURE_MAX_VALUE,
                value=TEMPERATURE_DEFAULT_VALUE,
                step=TEMPERATURE_STEP,)

    # Define a dictionary with the function names and their respective functions
    functions = [
        "home",
        "Chat with CSV",
        "Summarize CSV",
        "Analyze CSV",
    ]
    
    #st.subheader("Select any generator👇")
    # Create a selectbox with the function names as options
    selected_function = st.selectbox("Select a functionality", functions)
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name)
    elif selected_function == "Summarize CSV":
        summary(model_name=model_name, temperature=temperature, top_p=top_p)
    elif selected_function == "Analyze CSV":
        analyze(temperature=temperature, model_name=model_name)
    else:
        st.warning("You haven't selected any AI Functionality!!")
    

    

if __name__ == "__main__":
    main()
