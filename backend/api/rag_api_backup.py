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



logging.basicConfig(level=logging.CRITICAL)

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
        ("human", "{question}"),
    ]
)

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

print("prompt: ",prompt)
#question = "\nWho signs the Project Completion Letter?\n"
retriever = docsearch.as_retriever(search_kwargs={'k': 3, 'temperature': 0.2, 'top_p': 0.9})

retrieval_qa_chain_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,return_source_documents=True, input_key="question")
print("\ninput_data: ",retrieval_qa_chain_with_sources,"\n")
#result = retrieval_qa_chain_with_sources({"question": question})
#ai_response=dumps(result, pretty=True)
#ai_response=json.loads(ai_response)
#print("\nai_response: ",ai_response,"\n")


#chain = prompt | qa_with_sources | output_parser
#chain =  RunnablePassthrough.assign(retriever=docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9}))| prompt | qa_with_sources |output_parser
#chain =  RunnablePassthrough.assign(retriever=docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9}))| prompt | qa_with_sources
#chain =  RunnablePassthrough.assign(retriever=docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9}))|format_docs| prompt | qa_with_sources |output_parser

#context = itemgetter("question") |docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9})| format_docs
#first_step = RunnablePassthrough.assign(context=context)
#chain = first_step | prompt | qa_with_sources |output_parser


#chain=prompt|qa_with_sources|llm |output_parser




"""

print("Inputs to chain_with_history:", {"question": prompt})

print("SK-RunnablePassthrough(): ",RunnablePassthrough())
chain = (
       {"context": docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9}), "question": RunnablePassthrough()}
       #| prompt
       | retrieval_qa_chain_with_sources
       | StrOutputParser()
       )

"""

chain=prompt|llm |output_parser # <--- Working one-->


#print("\nchain",chain,"\n")
print("\nchain",chain,"\n")

# Integrate with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda user_session_id: DynamoDBChatMessageHistory(
        table_name=session_table_name, session_id=user_session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)
print("chain_with_history: ",chain_with_history)