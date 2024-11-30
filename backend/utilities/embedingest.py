#Import all required libraries.
import json
import boto3
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrockConverse
from langchain_community.document_loaders import Docx2txtLoader
from opensearchpy import OpenSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import botocore
import time
from langchain_community.llms import Bedrock
from langchain.load.dump import dumps
import warnings
import json
import os
import sys
from . import bedrockclient
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


warnings.filterwarnings('ignore')
module_path = ".."
#print('os.path.abspath(module_path): ',os.path.abspath(module_path))
sys.path.append(os.path.abspath(module_path))


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

# OpenSearch End Point and Collection Name

OPENSEARCH_ENDPOINT = "https://6yhu0ulyk52ugfx8eu5i.us-east-1.aoss.amazonaws.com"
INDEX_NAME = "ashu-open-search-vector-db"

#Create bedrock client instance function.

def get_bedrock_client():

    boto3_bedrock = bedrockclient.get_bedrock_client(
        #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )
    return boto3_bedrock

#Create bedrock embedding instance function.

def get_bedrock_embedding_model():
    boto3_bedrock=get_bedrock_client()
    bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
    return bedrock_embeddings

#Create titan embedding instance function.

def get_titan_embedding_model():
    embedding_model=BedrockEmbeddings(
    credentials_profile_name= 'default',
    model_id='amazon.titan-embed-text-v1')
    return embedding_model

# create the Anthropic Model

def get_bedrock_anthropic_claude_llm():
    boto3_bedrock=get_bedrock_client()
    llm = Bedrock(
    model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample": 200,"temperature": 0.2,"top_p": 0.9}
    )
    return llm
# create the Anthropic Model using ChatBedrockConverse

def get_anthropic_claude_llm():
    anthropic_claude_llm=model = ChatBedrockConverse(
        model="anthropic.claude-3-haiku-20240307-v1:0",
      max_tokens=300,
      temperature=0.1,
      top_p=.09,
      stop_sequences=["\n\nHuman"],
      verbose=True
      )
    return anthropic_claude_llm

def store_opensearch_embeddings(split_data,embedding_model):
    vector_store_name = 'ashu-open-search-vector-db'
    index_name = "ashu-open-search-vector-db-index"
    encryption_policy_name = "easy-ashu-open-search-vector-db"
    network_policy_name = "easy-ashu-open-search-vector-db"
    access_policy_name = 'easy-ashu-open-search-vector-db'

    aoss_client = boto3.client('opensearchserverless')
    collection = aoss_client.batch_get_collection(names=[vector_store_name])
    host = collection['collectionDetails'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    #print("SURESH-OpenSource-Host", host)

    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, os.environ.get("AWS_DEFAULT_REGION", None), service)

    docsearch = OpenSearchVectorSearch.from_documents(
        split_data,
        embedding_model,
        opensearch_url=host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=index_name,
        engine="faiss",
    )
    return docsearch


        
def process_file(bucket_name, key,embedding_model):
    """Process the file: chunk, embed, and store."""
    # Download the file
    local_path = f"/tmp/{key.split('/')[-1]}"
    s3.download_file(bucket_name, key, local_path)
    print('local_path: ', local_path)

    # Read file content
    with open(local_path, "r",encoding='UTF-8') as file:
        print(file)
        loader = Docx2txtLoader(local_path)
        document= loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=200,
        )
        split_data = text_splitter.split_documents(document)
        avg_doc_length = lambda split_data: sum([len(doc.page_content) for doc in split_data])
        len(split_data)
        avg_char_count_pre = avg_doc_length(split_data)
        avg_char_count_post = avg_doc_length(document)
        print(f"Average length among {len(document)} documents loaded is {avg_char_count_pre} characters.")
        print(f"After the split we have {len(split_data)} documents more than the original {len(document)}.")
        print(
            f"Average length among {len(split_data)} documents (after split) is {avg_char_count_post} characters."
        )

    try:
        sample_embedding = np.array(embedding_model.embed_query(split_data[0].page_content))
        modelId = embedding_model.model_id
        print("Embedding model Id :", modelId)
        print("Sample embedding of a document chunk: ", sample_embedding)
        print("Size of the embedding: ", sample_embedding.shape)

    except ValueError as error:
        if  "AccessDeniedException" in str(error):
                print(f"\x1b[41m{error}\
                \nTo troubeshoot this issue please refer to the following resources.\
                \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
                class StopExecution(ValueError):
                    def _render_traceback_(self):
                        pass
                raise StopExecution        
        else:
            raise error    

    opensearch_db_index = store_opensearch_embeddings(split_data,embedding_model)


def main():
    print('\nHello! Suresh!! Welcome to RAG Experimentaion...!\n')
    bucket_name='ashu-data'
    key='test-file.docx'
    model = get_bedrock_embedding_model();
    # Process the uploaded file
    #process_file(bucket_name,key,model)

    query = "Is it possible that I get sentenced to jail due to failure in filings?"
    query = "Who signs the Project Completion Letter?"
    vector_store_name = 'ashu-open-search-vector-db'
    index_name = "ashu-open-search-vector-db-index"
    embdedding_model=get_bedrock_embedding_model()

    aoss_client = boto3.client('opensearchserverless')
    collection = aoss_client.batch_get_collection(names=[vector_store_name])
    aoss_host_name = collection['collectionDetails'][0]['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    #print("SURESH-OpenSource-Host", aoss_host_name)

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
    llm=get_bedrock_anthropic_claude_llm()

    '''
    results = docsearch.similarity_search(query, k=3)  # our search query  # return 3 most relevant docs
    print("\n")
    print("\nSimilariry Search...\n")
    print(dumps(results, pretty=True))

    
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    #qa.run(query)
    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 3,'temperature': 0.2,'top_p': 0.9}),return_source_documents=True)
    print("\n")
    print("\n")   
    
    print(dumps(qa_with_sources(query), pretty=True))
    ai_response=dumps(qa_with_sources(query), pretty=True)
    ai_response=json.loads(ai_response)
    print("Human Query: ", ai_response["query"])
    print("\nAI Response: ", ai_response["result"])
    print("\n")  
    print("\n")   
    '''

    prompt_template = """Human: Use the following pieces of context to provide a concise answer in English to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_prompt = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    query = " Who signs the Project Completion Letter?"
    result = qa_prompt({"query": query})
    #print(result)
    ai_response=dumps(result, pretty=True)
    ai_response=json.loads(ai_response)
    print("Human Query: ", ai_response["query"])
    print("\nAI Response: ", ai_response["result"])
    
if __name__ == "__main__":
    main()