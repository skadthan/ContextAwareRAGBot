import boto3
client = boto3.client('sts')

def get_iam_user_id():    
    # Get the caller identity
    identity = client.get_caller_identity()
    
    # Extract and return the User ID
    user_id = identity['UserId']
    return user_id

# Get UserId for sessionId
user_id = get_iam_user_id()
print(user_id)