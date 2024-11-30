from aws_cdk import Stack
from constructs import Construct

from iac.cdk.create_opensearch_serverless_collection import CreateOpenSearchCollection
from iac.cdk.create_sagemaker_notebook import CreateSageMakerNotebook
from iac.cdk.create_sagemaker_notebook_iam_role import CreateSageMakerNotebookRole


class RagChatBotStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs) 

        self.collection_name = 'ashu-open-search-vector-collection'
        self.sagemaker_notebook_instance_type = 'ml.t3.medium'
        self.git_repo = "https://github.com/skadthan/rag-chat-bot.git"
        self.current_user_arn = self.node.try_get_context("current_user_arn")
        print('self.current_user_arn', self.current_user_arn)

        sagemaker_notebook_role = CreateSageMakerNotebookRole(self, "CreateSageMakerNotebookRole", {})
        opensearch_collection = CreateOpenSearchCollection(self, "CreateOpenSearchCollection", {
            "collection_name": self.collection_name,
            "notebook_role_arn": sagemaker_notebook_role.role_arn,
            "current_user_arn": self.current_user_arn
        })

        sagemaker_notebook = CreateSageMakerNotebook(self, "CreateSageMakerNotebook", {
            "collection_arn": opensearch_collection.collection_arn,
            "notebook_instance_type": self.sagemaker_notebook_instance_type,
            "notebook_role": sagemaker_notebook_role.role,
            "notebook_role_arn": sagemaker_notebook_role.role_arn,
            "git_repo": self.git_repo
        })
        
        sagemaker_notebook.node.add_dependency(sagemaker_notebook_role)
        sagemaker_notebook.node.add_dependency(opensearch_collection)