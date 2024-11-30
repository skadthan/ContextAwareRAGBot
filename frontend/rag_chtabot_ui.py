from backend.api import rag_api as chat_api
import streamlit as st


st.title("Hi, Ashu & Ananaya! Welcome to Claude AI Chatbot! :sunglasses:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Load the stored messages from DynamoDB
    #stored_messages = chat_api.history.messages  # Retrieve all stored messages
    stored_messages = chat_api.get_session_history(chat_api.user_session_id)  # Correctly retrieve messages

    # Populate the session state with the retrieved messages
    for msg in stored_messages:
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type to talk to Claude AI Chatbot!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

  # Generate assistant response using Bedrock LLM and LangChain
    config = {"configurable": {"session_id": chat_api.user_session_id}}
    #print("UI-Prompt:", prompt)
    #print("UI Input (input):", {"input": prompt})
    #breakpoint()
    
    response = chat_api.chain_with_history.invoke({"input": str(prompt)}, config=config)
    
    #response = chat_api.chain_with_history.invoke(prompt, config=config)

    # Iterate over the context and extract metadata and page_content
    for doc in response['context']:
        source = doc.metadata.get('source', 'Unknown')  # Fetch 'source' metadata
        content = doc.page_content  # Fetch the page content
        print(f"Source: {source}")
        print(f"Content: {content[:100]}...")  # Print the first 100 characters of content for brevity
        print("-" * 50)
        source = "Document Ref: "+source
    response=response.get('answer')

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(source)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})