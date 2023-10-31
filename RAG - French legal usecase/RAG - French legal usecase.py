#!/usr/bin/env python
# coding: utf-8

# # RAG application with Azure Open AI & Azure Cognitive Search
# ## French legal usecase
# 
# <img src="https://github.com/retkowsky/images/blob/master/azure_openai_logo.png?raw=true" width=400>
# 
# ### Objective
# Let's build an application that will analyse **legal PDF documents using a RAG application.**
# 
# **Retrieval-Augmented Generation (RAG)** can exhibit variability in its implementation, but at a fundamental level, employing RAG within an AI-driven application involves the following sequential steps:
# 
# - The user submits a query or question.
# - The system initiates a search for pertinent documents that hold the potential to address the user's query. These documents are often comprised of proprietary data and are maintained within a document index.
# - The system formulates an instruction set for the Language Model (LLM) that encompasses the user's input, the identified relevant documents, and directives on how to utilize these documents to respond to the user's query effectively.
# - The system transmits this comprehensive prompt to the Language Model.
# - The Language Model processes the prompt and generates a response to the user's question, drawing upon the context provided. This response constitutes the output of our system.
# 
# ### Steps
# - Uploading PDF documents into an Azure Cognitive Search Index
# - Use of some Azure Cognitive Search queries to get some answers
# - Use a GPT model to analyse the answer (summmary, keywords generation)
# - Get the text from the document and the reference to validate the proposed answer
# - Chatbot experience using Azure Open AI to ask questions and get results provided by AI with references
# 
# ### Process
# <img src="https://github.com/retkowsky/images/blob/master/rag.png?raw=true" width=800>

# In[1]:


# %pip install azure-search-documents==11.4.0b8


# In[2]:


#import azure.cognitiveservices.speech as speechsdk
import datetime
import gradio as gr
import json
import langchain
import openai
import os
import sys
import time

from azure_rag import (
    get_storage_info,
    get_stats_from_pdf_file,
    delete_index,
    ask_gpt,
    index_status,
    index_stats,
    openai_text_embeddings,
    chunking,
    upload_docs,
    similarity_comparison,
    azure_openai,
    ask_your_own_data,
)
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch


# In[3]:


print(f"Today: {datetime.datetime.today().strftime('%d-%b-%Y %H:%M:%S')}")


# In[4]:


get_storage_info()


# ## 1. Settings

# In[5]:


print(f"Python version: {sys.version}")
print(f"OpenAI version: {openai.__version__}")
print(f"Langchain version: {langchain.__version__}")


# In[6]:


load_dotenv("azure.env")

# Azure Open AI
openai.api_type: str = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")

# Azure Cognitive Search
azure_cs_endpoint = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT")
azure_cs_key = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")


# In[7]:


# Azure Open AI models (Should be deployed on your Azure OpenAI studio)
embed_model: str = "text-embedding-ada-002"  # Embedding model
gptmodel: str = "gpt-35-turbo-16k"  # GPT Model

# Azure Cognitive search index
index_name: str = "french-penal-code-rag"


# In[8]:


print(
    f"We will use {embed_model} as the embedding model and {gptmodel} as the Azure Open AI model"
)


# In[9]:


print(f"We will create the Azure Cognitive Search index: {index_name}")


# ## 2. PDF documents

# In[10]:


PDF_DIR = "documents"

pdf_files = [file for file in os.listdir(PDF_DIR) if file.lower().endswith(".pdf")]
pdf_files


# In[11]:


get_ipython().system('ls $PDF_DIR/*.pdf -lh')


# ## 3. Loading the French PDF documents

# We will chunk our PDF documents, do the embeddings and save the content into Azure Cognitive Search

# In[15]:


embed_model


# In[13]:


# Embeddings engine
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(engine=embed_model)


# In[14]:


# Azure Cognitive Search as the vector store
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_cs_endpoint,
    azure_search_key=azure_cs_key,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)


# ### Testing the models

# In[16]:


messages_history = []
messages_history = ask_gpt("Hello", messages_history, gptmodel)


# In[18]:


messages_history = ask_gpt("What can you do?", messages_history, gptmodel)


# ### Testing the embedded model

# In[19]:


prompt1 = "My name is James Bond"
prompt2 = "Ian Fleming"
prompt3 = "Azure Open AI"


# In[20]:


emb1 = openai_text_embeddings(prompt1, embed_model)
emb2 = openai_text_embeddings(prompt2, embed_model)
emb3 = openai_text_embeddings(prompt3, embed_model)


# In[21]:


emb1[:10]


# In[22]:


len(emb1)


# In[23]:


print("Similarity between:", prompt1, "and", prompt1)
similarity_comparison(emb1, emb1)


# In[24]:


print("Similarity between:", prompt1, "and", prompt2)
similarity_comparison(emb1, emb2)


# In[25]:


print("Similarity between:", prompt1, "and", prompt3)
similarity_comparison(emb1, emb3)


# In[26]:


print("Similarity between:", prompt2, "and", prompt3)
similarity_comparison(emb2, emb3)


# ### Processing the PDF docs

# In[28]:


get_ipython().system('ls $PDF_DIR -lh')


# In[29]:


for i in range(len(pdf_files)):
    # Getting stats for each file
    filepath = os.path.join(PDF_DIR, pdf_files[i])
    get_ipython().system('ls $filepath -lh')
    get_stats_from_pdf_file(filepath)


# In[30]:


start = time.time()
print("Documents processing...\n")

for i in range(len(pdf_files)):
    print(f"Uploading file {i + 1}")
    # Chunking and loading into the Azure Cognitive Search index
    upload_docs(PDF_DIR, pdf_files[i], vector_store)
    time.sleep(5)

elapsed = time.time() - start
print("\nDone")
print(
    "Elapsed time: "
    + time.strftime(
        "%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:15], time.gmtime(elapsed)
    )
)


# ## 4. Azure Cognitive Search index status

# Our index is available and ready to use

# In[31]:


print(f"Azure Cognitive Search index = {index_name}")


# In[32]:


index_status(index_name, azure_cs_endpoint, azure_cs_key)


# In[33]:


document_count, storage_size = index_stats(index_name, azure_cs_endpoint, azure_cs_key)


# In[34]:


print(f"Number of documents in the index = {document_count}")
print(f"Size of the index = {round(storage_size / (1024 * 1024), 2)} MB")


# ## 5. Search with Azure Open AI

# In[35]:


get_ipython().run_cell_magic('javascript', 'Python', 'OutputArea.auto_scroll_threshold = 9999')


# > This is to maximize the display output

# ### 5.1 Testing

# We can do some quick tests

# In[36]:


print(f"Azure OpenAI model to use: {gptmodel}")


# In[37]:


print(azure_openai("What time is it?", gptmodel))


# In[38]:


print(azure_openai("Who are you?", gptmodel))


# In[39]:


print(azure_openai("Qui es-tu?", gptmodel))


# In[40]:


print(azure_openai("What can you do for me?", gptmodel))


# ### 5.2 Ask your own data 

# We can ask any questions related with the documents we have processed. You can ask in any language. Results are also saved into .docx documents

# In[41]:


query = "What are the French laws regarding corruption?"

ask_your_own_data(query, vector_store, gptmodel, topn=3)


# In[42]:


query = "Quelle est la peine encourue pour un vol ?"

ask_your_own_data(query, vector_store, gptmodel, topn=3)


# In[43]:


query = "Quelles sont les peines encourues pour un vol ?"

ask_your_own_data(query, vector_store, gptmodel, topn=3)


# In[44]:


query = "Quelles sont les missions du procureur général ?"

ask_your_own_data(query, vector_store, gptmodel, topn=3)


# In[45]:


query = "¿Qué es el artículo 131-39 del Código Penal?"

ask_your_own_data(query, vector_store, gptmodel, topn=1)


# In[46]:


query = "What is a crime against the nation?"

ask_your_own_data(query, vector_store, gptmodel, topn=1)


# In[47]:


query = "What are the penalties for forgery?"

ask_your_own_data(query, vector_store, gptmodel, topn=1)


# ## 6. Gradio webapp

# Let's build a quick webapp with gradio to test our RAG application

# In[54]:


def legal_rag_function(query):
    """
    RAG function with Azure Cognitive Search and Azure Open AI
    Input: query
    Output: result list with the results for a gradio webapp
    """
    # Warning message
    msg = "\n\n[Note] This summary is generated by an AI (powered by Azure Open AI). Examine and use carefully."
    
    # Calling Azure Cognitive Search
    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=5,
        score_threshold=0.5,
    )

    fulltext_list = []

    for idx in range(5):
        reference = results[idx][0].__dict__["page_content"]
        fulltext_list.append(reference)

    # Get the AOAI summary
    fulltext = "".join(fulltext_list)
    answer = azure_openai(fulltext, gptmodel)
    
    # Lists declaration
    results_list = []
    source_list = []
    confidence_list = []

    # Output generation
    for i in range(5):
        # Get Azure Cognitive Search result
        reference = results[i][0]
        ref = reference.__dict__
        ref = ref["page_content"]

        # Get source information
        source = reference.metadata
        doc_source = source["source"]
        page_source = source["page"]
        confidence = results[i][1]

        source_ref = f"Source: {doc_source} Page: {page_source}"
        source_list.append(source_ref)
        confidence_list.append(confidence)

        # Adding all the results into a single list
        results_list.append(ref)
        results_list.append(source_list[i])
        results_list.append(confidence_list[i])

    results_list.insert(0, answer + msg)  # Insert the answer on the top of the list

    return results_list


# ### Webapp function definition

# In[57]:


image_url = "https://github.com/retkowsky/images/blob/master/legal.jpg?raw=true"
image = "<center> <img src= {} width=200px></center>".format(image_url)
header = "RAG application with Azure Open AI & Azure Cognitive Search (French penal code example)"

samples = [
    "Identity checks",
    "What is the definition of a crime against humanity?",
    "What are the penalties for fraud?",
    "Article 131-39 of the penal code",
    "Artículo 131-39 del Código Penal",
    "Quid des frais de justice ?",
    "Quelles sont les missions du procureur général ?",
]

inputs = gr.Text(type="text", label="Question:", lines=5, max_lines=10)

outputs = [
    gr.Text(label="Azure Open AI summary"),
    gr.Text(label="1. Reference"),
    gr.Text(label="1. Document source"),
    gr.Number(label="1. Confidence"),
    gr.Text(label="2. Reference"),
    gr.Text(label="2. Document source"),
    gr.Number(label="2. Confidence"),
    gr.Text(label="3. Reference"),
    gr.Text(label="3. Document source"),
    gr.Number(label="3. Confidence"),
    gr.Text(label="4. Reference"),
    gr.Text(label="4. Document source"),
    gr.Number(label="4. Confidence"),
    gr.Text(label="5. Reference"),
    gr.Text(label="5. Document source"),
    gr.Number(label="5. Confidence"),
]

theme = "freddyaboulton/test-blue"
# https://huggingface.co/spaces/gradio/theme-gallery

legal_rag_webapp = gr.Interface(
    fn=legal_rag_function,
    inputs=inputs,
    outputs=outputs,
    description=image,
    title=header,
    examples=samples,
    theme=theme,
)


# ### Running the webapp

# In[58]:


legal_rag_webapp.launch(share=True)


# ## 7. Post Processing

# We can delete our index if needed

# In[ ]:


index_status(index_name, azure_cs_endpoint, azure_cs_key)


# In[ ]:


delete_index(index_name, azure_cs_endpoint, azure_cs_key)


# In[ ]:


index_status(index_name, azure_cs_endpoint, azure_cs_key)


# > End of notebook

# In[ ]:




