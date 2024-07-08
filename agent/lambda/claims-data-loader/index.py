import json
import os
import boto3
import logging
import cfnresponse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

claims_table_name = os.environ.get('CLAIMS_TABLE_NAME')
region = os.environ.get('AWS_REGION')

dynamodb = boto3.client('dynamodb', region_name=region)

def to_dynamodb_attribute(value):
    if value is None:
        print(f"value 1 = {str(value)}")
        return {'NULL': True}
    elif isinstance(value, str):
        print(f"value 2 = {str(value)}")
        return {'S': value}
    elif isinstance(value, bool):
        print(f"value 3 = {str(value)}")
        return {'BOOL': value}
    elif isinstance(value, (int, float)):
        print(f"value 4 = {str(value)}")
        return {'N': str(value)}
    elif isinstance(value, dict):
        print(f"value 5 = {str(value)}")
        nested_attributes = {}
        for nested_key, nested_value in value.items():
            print(f"nested_value 6 = {str(nested_value)}")
            nested_attributes[nested_key] = to_dynamodb_attribute(nested_value)
        return {'M': nested_attributes}
    elif isinstance(value, list):
        print(f"value 7 = {str(value)}")
        list_attribute = []
        for item in value:
            print(f"item 8 = {str(item)}")
            list_attribute.append(to_dynamodb_attribute(item))
        return {'L': list_attribute}
    else:
        raise ValueError("Unsupported data type: {}".format(type(value)))

def handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    request_type = event.get('RequestType')
    if request_type == 'Create' or request_type == 'Update':
        try:
            with open('claims.json', 'r') as file:
                claims_data = json.load(file)
            
            items = []
            for claim in claims_data:
                print(f"claim = {str(claim)}")
                item = {}
                for key, value in claim.items():
                    item[key] = to_dynamodb_attribute(value)

                # Add claimId and policyId attributes
                item['claimId'] = {'S': claim['claimId']}
                item['policyId'] = {'S': claim['policyId']}
                
                items.append({'PutRequest': {'Item': item}})
            
            print(f"batch_write_item items = {str(items)}")
            response = dynamodb.batch_write_item(
                RequestItems={
                    claims_table_name: items
                }
            )
            
            logger.info("Batch write response: %s", json.dumps(response))
            cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData={})
        except Exception as e:
            logger.error("Failed to load data into DynamoDB table: %s", str(e))
            cfnresponse.send(event, context, cfnresponse.FAILED, responseData={"Error": str(e)})

    elif request_type == 'Delete':
        cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData={})

    return {
        'statusCode': 200,
        'body': json.dumps('Function execution completed successfully')
    }