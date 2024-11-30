from operator import itemgetter
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import logging
from langchain.load.dump import dumps
import json

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import boto3
import os
from requests_aws4auth import AWS4Auth

from backend.utilities import getiamuserid
from backend.utilities import embedingest

from langchain_core.runnables import RunnablePassthrough
import warnings

from io import StringIO
import sys
import textwrap
import os
from typing import Optional

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings('ignore')

# Specify the AWS profile name
os.environ["AWS_DEFAULT_REGION"] = boto3.Session().region_name 
session = boto3.Session(profile_name='default')

# Initialize awsauth, open search parameters, boto clients and llm model 
s3 = session.client('s3')
client = boto3.client('opensearchserverless')
service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                    region, service, session_token=credentials.token)

# Initialize the DynamoDB chat message history
session_table_name = "SessionTable"
user_session_id = getiamuserid.get_iam_user_id()  # You can make this dynamic based on the user session
history = DynamoDBChatMessageHistory(table_name=session_table_name, session_id=user_session_id)


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))

def format_docs(docs):
    #print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(user_session_id):
    history = DynamoDBChatMessageHistory(
        table_name=session_table_name,
        session_id=user_session_id,
    )
    return history.messages


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If the answer is not present in the context, just say you do not have enough context to answer. \
If the input is not present in the context, just say you do not have enough context to answer. \
If the question is not present in the context, just say you do not have enough context to answer. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# Create output parser to simplify the output
output_parser = StrOutputParser()

vector_store_name = 'ashu-open-search-vector-db'
index_name = "ashu-open-search-vector-db-index"
embdedding_model=embedingest.get_bedrock_embedding_model()

aoss_client = boto3.client('opensearchserverless')
collection = aoss_client.batch_get_collection(names=[vector_store_name])
aoss_host_name = collection['collectionDetails'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'

auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)

docsearch = OpenSearchVectorSearch(
    index_name=index_name,
    embedding_function=embdedding_model,
    opensearch_url=aoss_host_name,
    http_auth=auth,
    timeout = 100,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    engine="faiss",
)

    #LLM Integration
    #llm=get_anthropic_claude_llm()
llm=embedingest.get_bedrock_anthropic_claude_llm()

# Combine the prompt with the Bedrock LLM

#print("prompt: ",prompt)
#question = "\nWho signs the Project Completion Letter?\n"
retriever = docsearch.as_retriever(search_kwargs={'k': 3, 'temperature': 0.2, 'top_p': 0.9})

#retrieval_qa_chain_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=True, input_key="input")
#print("\ninput_data: ",retrieval_qa_chain_with_sources,"\n")


chain=prompt|llm |output_parser # <--- Working one-->

#print(f"Retriever: {retriever}")
#print(f"Prompt Variables: {prompt.input_variables}")


history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=prompt,  # Ensure prompt variables align
)

#print(f"Retriever: {history_aware_retriever}")

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#print(f"Chain Input: {rag_chain}")


#print("\nquestion_answer_chain",question_answer_chain,"\n")

#print("\nchain",chain,"\n")

# Integrate with message history
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda user_session_id: DynamoDBChatMessageHistory(
        table_name=session_table_name, session_id=user_session_id
    ),
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

#print("chain_with_history: ",chain_with_history)
