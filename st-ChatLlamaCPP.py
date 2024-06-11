import streamlit as st
import datetime
import os
from io import StringIO
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=90)
import tiktoken
from time import sleep

encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))

from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from llama_cpp import Llama

st.set_page_config(layout="wide", page_title="LlamaCPP AIO chat with documents")

@st.cache_resource 
def create_embeddings():   
# Set HF API token  and HF repo
    from langchain_community.embeddings import LlamaCppEmbeddings
    embpath = "models/all-MiniLM-L6-v2.F16.gguf"
    embeddings = LlamaCppEmbeddings(model_path=embpath)
    print('loading all-MiniLM-L6-v2.F16.gguf with LlamaCPP...')
    return embeddings

@st.cache_resource 
def create_chat():   
# Set HF API token  and HF repo
    from llama_cpp import Llama
    qwen05b = Llama(
                model_path='models/qwen2-0_5b-instruct-q8_0.gguf',
                n_gpu_layers=0,
                temperature=0.1,
                top_p = 0.5,
                n_ctx=8192,
                max_tokens=600,
                repeat_penalty=1.7,
                stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
                verbose=False,
                )
    print('loading qwen2-0_5b-instruct-q8_0.gguf with LlamaCPP...')
    return qwen05b

@st.cache_data
def create_vectorstore_fromTXT(filepath,embeddings):
    # load a TXT document
    stringio = StringIO(filepath.getvalue().decode("utf-8"))
    #loader = TextLoader(filepath)
    #Create a document and split into chuncks
    documents = stringio.read()
    text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    #create the vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore, documents

# FUNCTION TO LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory-KSDOCS.txt', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

#AVATARS
av_us = '🧑‍💻'  # './man.png'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = "✨"   #'./robot.png'

if "gentime" not in st.session_state:
    st.session_state.gentime = "none yet"
if "docfile" not in st.session_state:
    st.session_state.docfile = ''   
if "keyimagefile" not in st.session_state:
    st.session_state.keyimagefile = 0     
if "chatdocs" not in st.session_state:
    st.session_state.chatdocs = 0   
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []   
if "chatUImessages" not in st.session_state:
    st.session_state.chatUImessages = [{"role": "assistant", "content": "Hi there! I am here to assist you with this document. What do you want to know?"}]   
if "uploadedDoc" not in st.session_state:
    st.session_state.uploadedDoc = [] 
if "uploadedText" not in st.session_state:
    st.session_state.uploadedText = '' 
if "data_uri" not in st.session_state: 
    st.session_state.data_uri = [] 

st.markdown("# 💬🖼️ Talk to your Documents\n\n### *LlamaCPP Qwen+embeddings*\n\n\n")
st.markdown('\n---\n', unsafe_allow_html=True)
st.sidebar.image('https://i.ytimg.com/vi/3H8X8q_XIJU/maxresdefault.jpg')
llm = create_chat()
embeddings = create_embeddings()

file1=None
#image_btn = st.button('✨ **Start AI Magic**', type='primary')
def resetall():
    # tutorial to reset the to 0 the file_uploader from 
    # https://discuss.streamlit.io/t/clear-the-file-uploader-after-using-the-file-data/66178/4
    st.session_state.keyimagefile += 1
    st.session_state.chatdocs = 0
    st.session_state.chatUImessages = [{"role": "system", "content": "You are a Language Model trained to answer questions based solely on the provided text.",}]
    st.rerun()

#Function for QnA over a Context - the context is pure string text
def QwenQnA(messages,question,vectorstore,hits,maxtokens,model):
  """
  basic generation with Qwen-0.5b-Cha / any llama.cpp loaded model
  question -> string
  contesto -> string, parsed page_content from document objects
  maxtokens -> int, number of max tokens to generate
  model -> llama-cpp-python instance // here is Qwen-0.5b-Chat AI
  RETURNS question, output -> str
  """
  #docs_and_scores = vectorstore.similarity_search_with_score(question,k=3) #or .similarity_search_with_relevance_scores()
  retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": hits})
  docs = retriever.invoke(question)
  context = ''
  for i in docs:
    context += i.page_content
  contesto = context.replace('\n\n','')
  query = question
  import datetime
  start = datetime.datetime.now()
  template = f"""Answer the question based only on the following context:
[context]
{contesto}
[end of context]

Question: {query}

"""
  messages.append({"role": "user", "content": template})
  with console.status("Qwen-0.5b-Chat AI is working ✅✅✅ ...",spinner="dots12"):
    output = model.create_chat_completion(
                    messages=messages,
                    max_tokens=maxtokens,
                    stop=["</s>","[/INST]","/INST",'<|eot_id|>','<|end|>'],
                    temperature = 0.1,
                    repeat_penalty = 1.4)
    delta = datetime.datetime.now() - start
    console.print(f"[bright_green bold on black]Question: {query}")
    console.print(output["choices"][0]["message"]["content"])
    console.print(f"Completed in: [bold bright_red]{delta}")

    return output["choices"][0]["message"]["content"], docs  

st.sidebar.write("## Upload a Document :gear:")
st.markdown('\n\n')
message1 = st.sidebar.empty()
message11 = st.sidebar.empty()
message2 = st.sidebar.empty()
message2.write(f'**:green[{st.session_state.gentime}]**')
message3 = st.empty()

# Upload the audio file
file1 = st.sidebar.file_uploader("Upload a text document", 
                                    type=["txt", "md"],accept_multiple_files=False, 
                                    key=st.session_state.keyimagefile)
print(file1)
#gentimetext = st.sidebar.empty()
reset_btn = st.sidebar.button('🧻✨ **Reset Document** ', type='primary')
rtrvDocs = st.sidebar.empty()

if file1:
    st.session_state.chatdocs = 1
    st.session_state.docfile = file1
    message1.write('Docfile file selected!')
    #stringio = StringIO(file1.getvalue().decode("utf-8"))
    with open(file1.name, mode='wb') as w:
        w.write(file1.getvalue())
    loader = TextLoader(file1.name)
    #Create a document and split into chuncks
    documents = loader.load()#stringio.read()
    text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    #create the vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    st.session_state.uploadedDoc = vectorstore 
    st.session_state.uploadedText = texts
    #st.session_state.uploadedDoc.save('temp.jpg')
    # https://stackoverflow.com/questions/52411503/convert-image-to-base64-using-python-pil
    # https://huggingface.co/docs/api-inference/detailed_parameters
    #st.session_state.data_uri =file('temp.jpg')
    message11.write('Embeddings OK. Ready to **CHAT**')        
    if reset_btn:
        resetall()
        try:
            st.session_state.uploadedDoc = [] 
            st.session_state.uploadedText = '' 
        except:
            pass

    # Display chat messages from history on app rerun
    for message in st.session_state.chatUImessages:
        if message["role"] == "user":
            with st.chat_message(message["role"],avatar=av_us):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"],avatar=av_ass):
                st.markdown(message["content"])
    # Accept user input
    if myprompt := st.chat_input("What is this?"): #,key=str(datetime.datetime.now())
        # Add user message to chat history
        st.session_state.messages = [
                    st.session_state.data_uri,
                    myprompt
                ]
        st.session_state.chatUImessages.append({"role": "user", "content": myprompt})
        # Display user message in chat message container
        with st.chat_message("user", avatar=av_us):
            st.markdown(myprompt)
            usertext = f"user: {myprompt}"
            writehistory(usertext)
            # Display assistant response in chat message container
        with st.chat_message("assistant",avatar=av_ass):
            message_placeholder = st.empty()
            docs_placeholder = st.empty()
            with st.spinner("Thinking..."):
                full_response = ""
                start = datetime.datetime.now()
                result, retrieved = QwenQnA(st.session_state.chatUImessages,myprompt,st.session_state.uploadedDoc, 4,500,llm)
                for chunk in result:
                    if full_response == '':
                        full_response=chunk
                        message_placeholder.markdown(full_response + "🔷")
                    else:
                        try:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "🔷")
                            sleep(0.018)
                        except:
                            pass                                           
            st.session_state.data_uri = retrieved
            rtrvDocs.write(st.session_state.data_uri)
            message_placeholder.markdown(full_response)
            docs_placeholder.write(st.session_state.data_uri)
            st.session_state.gentime = datetime.datetime.now() - start 
            message2.write(f'**:green[{str(st.session_state.gentime)}]**') 
            print(full_response)
            asstext = f"assistant: {full_response}"
            writehistory(asstext)       
            st.session_state.chatUImessages.append({"role": "assistant", "content": full_response})

if  not file1:
    message3.warning("  Upload a text document", icon='⚠️')

