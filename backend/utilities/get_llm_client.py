from langchain_aws import ChatBedrockConverse

def claude_llm():
    claude_llm=model = ChatBedrockConverse(
        model="anthropic.claude-3-haiku-20240307-v1:0",
      max_tokens=300,
      temperature=0.1,
      top_p=.09,
      stop_sequences=["\n\nHuman"],
      verbose=True
      )
    return claude_llm
