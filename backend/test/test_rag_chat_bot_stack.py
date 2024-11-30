import aws_cdk as core
import aws_cdk.assertions as assertions

from backend.iac import RagChatBotStack

# example tests. To run these tests, uncomment this file along with the example
# resource in opensearch_cdk/opensearch_cdk_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = RagChatBotStack(app, "test-rag-chat-bot-cdk")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })