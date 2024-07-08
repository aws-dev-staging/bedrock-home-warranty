import re
import os
import json
import time
import boto3
import pdfrw
import difflib
import logging
import datetime
import dateutil.parser

from chat import Chat
from insurance_agent import InsuranceAgent
from boto3.dynamodb.conditions import Key
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain

# Create reference to DynamoDB tables and S3 bucket
users_table_name = os.environ['USERS_TABLE_NAME']
claims_table_name = os.environ['CLAIMS_TABLE_NAME']
insurance_quote_requests_table_name = os.environ['INSURACE_QUOTE_REQUESTS_TABLE_NAME']
s3_artifact_bucket = os.environ['S3_ARTIFACT_BUCKET_NAME']

# Instantiate boto3 clients and resources
boto3_session = boto3.Session(region_name=os.environ['AWS_REGION'])
dynamodb = boto3.resource('dynamodb',region_name=os.environ['AWS_REGION'])
s3_client = boto3.client('s3',region_name=os.environ['AWS_REGION'],config=boto3.session.Config(signature_version='s3v4',))
s3_object = boto3.resource('s3')
bedrock_client = boto3_session.client(service_name="bedrock-runtime")

# --- Lex v2 request/response helpers (https://docs.aws.amazon.com/lexv2/latest/dg/lambda-response-format.html) ---

def elicit_slot(session_attributes, active_contexts, intent, slot_to_elicit, message):
    """
    Constructs a response to elicit a specific Amazon Lex intent slot value from the user during conversation.
    """
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ElicitSlot',
                'slotToElicit': slot_to_elicit 
            },
            'intent': intent,
        },
        'messages': [{
            "contentType": "PlainText",
            "content": message,
        }]
    }

    return response

def elicit_intent(intent_request, session_attributes, message):
    """
    Constructs a response to elicit the user's intent during conversation.
    """
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [
            {
                'contentType': 'PlainText', 
                'content': message
            },
            {
                'contentType': 'ImageResponseCard',
                'imageResponseCard': {
                    "buttons": [
                        {
                            "text": "Request Home Quote",
                            "value": "Home"
                        },
                        {
                            "text": "Request Auto Quote",
                            "value": "Auto"
                        },
                        {
                            "text": "Request Life Quote",
                            "value": "Life"
                        },
                        {
                            "text": "Ask GenAI",
                            "value": "What kind of questions can the assistant answer?"
                        }
                    ],
                    "title": "How can I help you?"
                }
            }     
        ]
    }

    return response

def delegate(session_attributes, active_contexts, intent, message):
    """
    Delegates the conversation back to the system for handling.
    """
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Delegate',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response

def build_slot(intent_request, slot_to_build, slot_value):
    """
    Builds a slot with a specified slot value for the given intent_request.
    """
    intent_request['sessionState']['intent']['slots'][slot_to_build] = {
        'shape': 'Scalar', 'value': 
        {
            'originalValue': slot_value, 'resolvedValues': [slot_value], 
            'interpretedValue': slot_value
        }
    }

def build_validation_result(isvalid, violated_slot, message_content):
    """
    Constructs a validation result indicating whether a slot value is valid, along with any violated slot and an accompanying message.
    """
    return {
        'isValid': isvalid,
        'violatedSlot': violated_slot,
        'message': message_content
    }
    
# --- Utility helper functions ---

def isvalid_number(value):
    # regex to match a valid numeric string without leading '-' for negative numbers or value "0"
    return bool(re.match(r'^(?:[1-9]\d*|[1-9]\d*\.\d+|\d*\.\d+)$', value))

def isvalid_date(year):
    try:
        year_int = int(year)
        current_year = int(datetime.datetime.now().year)
        print(f"Year input: {year_int}, Current year: {current_year}")  # Debugging output
        # Validate if the year is within a reasonable range
        if year_int <= 0 or year_int > current_year:
            return False
        return True
    except ValueError as e:
        print(f"isvalid_date error: {e}")
        return False

def isvalid_slot_value(value, slot_value_list): # Need to adjust
    # Adjust this threshold as needed
    similarity_threshold = 0.65

    # Calculate similarity using difflib
    similarity_scores = [difflib.SequenceMatcher(None, value.lower(), ref_value).ratio() for ref_value in slot_value_list]

    print(f"isvalid_slot_value similarity_scores: {similarity_scores}")
    # Check if the word is close to 'yes' or 'no' based on similarity threshold
    return any(score >= similarity_threshold for score in similarity_scores)

def create_presigned_url(bucket_name, object_name, expiration=600):
    """
    Generate a presigned URL for the S3 object.
    """
    try:
        response = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"Error creating S3 presigned URL: {e}")

    return None

def try_ex(value):
    """
    Safely access slots dictionary values.
    """
    if value and value.get('value'):
        return value['value'].get('interpretedValue') or value['value'].get('originalValue')
    return None

def get_user_by_policy_id(policyId):
    """
    Retrieves user information based on the provided policyId using a GSI.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Set up the query parameters for the GSI
        params = {
            'IndexName': 'PolicyIdIndex',
            'KeyConditionExpression': 'policyId = :pid',
            'ExpressionAttributeValues': {
                ':pid': policyId
            }
        }

        # Execute the query and get the result
        response = users_table.query(**params)

        # Check if any items were returned
        if response['Count'] > 0:
            return response['Items']
        else:
            print("No user found with the given policyId")

    except Exception as e:
        print(f"Error retrieving user by policyId: {e}")
    
    return None 

# --- Intent fulfillment functions ---

def isvalid_pin(username, pin):
    """
    Validates the user-provided PIN using a DynamoDB table lookup.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Query the table using the partition key
        response = users_table.query(
            KeyConditionExpression=Key('userName').eq(username)
        )

        # Iterate over the items returned in the response
        if len(response['Items']) > 0:
            pin_to_compare = int(response['Items'][0]['pin'])
            # Check if the password in the item matches the specified password
            if pin_to_compare == int(pin):
                return True

        print("PIN did not match")
        return False

    except Exception as e:
        print(f"Error validating PIN: {e}")
        return e

def isvalid_username(username):
    """
    Validates the user-provided username exists in the 'claims_table_name' DynamoDB table.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'userName = :c',
            'ExpressionAttributeValues': {
                ':c': username
            }
        }

        # Execute the query and get the result
        response = users_table.query(**params)     

        # Check if any items were returned
        if response['Count'] != 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error validating username: {e}")
        return e

def validate_pin(intent_request, username, pin):
    """
    Elicits and validates user input values for username and PIN. Invoked as part of 'verify_identity' intent fulfillment.
    """
    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
        session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
        session_attributes['UserName'] = username
        intent_request['sessionState']['sessionAttributes']['UserName'] = username
    else:
        return build_validation_result(
            False,
            'UserName',
            'Our records indicate there are no accounts belonging to that username. Please try again.'
        )

    if pin is not None:
        if  not isvalid_pin(username, pin):
            return build_validation_result(
                False,
                'Pin',
                'You have entered an incorrect PIN. Please try again.'.format(pin)
            )
    else:
        message = "Thank you for choosing AnyCompany, {}. Please confirm your 4-digit PIN before we proceed.".format(username)
        return build_validation_result(
            False,
            'Pin',
            message
        )

    return {'isValid': True}

def verify_identity(intent_request):
    """
    Performs dialog management and fulfillment for username verification.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting.
    2) Use of sessionAttributes {UserName} to pass information that can be used to guide conversation.
    """
    slots = intent_request['sessionState']['intent']['slots']
    pin = try_ex(slots['Pin'])
    username=try_ex(slots['UserName'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    # Validate any slots which have been specified. If any are invalid, re-elicit for their value
    intent_request['sessionState']['intent']['slots']
    validation_result = validate_pin(intent_request, username, pin)
    session_attributes['UserName'] = username

    if not validation_result['isValid']:
        slots = intent_request['sessionState']['intent']['slots']
        slots[validation_result['violatedSlot']] = None

        return elicit_slot(
            session_attributes,
            active_contexts,
            intent_request['sessionState']['intent'],
            validation_result['violatedSlot'],
            validation_result['message']
        )
    else:
        if confirmation_status == 'None':
            # Query DDB for user information before offering intents
            users_table = dynamodb.Table(users_table_name)

            try:
                # Query the table using the partition key
                response = users_table.query(
                    KeyConditionExpression=Key('userName').eq(username)
                )

                # Customize message based on coverage type
                message = ""
                items = response['Items']
                for item in items:
                    coverage_type = item.get('coverageType', None)
                    if coverage_type == 'Home':
                        message += f"Your home insurance policy provides comprehensive coverage for your property located at {item['propertyAddress']['street']} in {item['propertyAddress']['city']}, {item['propertyAddress']['state']} {item['propertyAddress']['zip']}. "
                        message += f"The policy started on {item['policyStartDate']} and ends on {item['policyEndDate']}. "
                        message += f"Your deductible amount is ${item['deductibleAmount']:,}."
                    elif coverage_type == 'Auto':
                        message += f"Your auto insurance policy covers your {item['vehicleMake']} {item['vehicleModel']} {item['vehicleYear']} with comprehensive coverage. "
                        message += f"The policy started on {item['policyStartDate']} and ends on {item['policyEndDate']}. "
                        message += f"Your insured amount is ${item['insuredAmount']:,}."
                    elif coverage_type == 'Life':
                        message += f"Your life insurance policy provides coverage for you with an insured amount of ${item['insuredAmount']:,}. "
                        message += f"The policy started on {item['policyStartDate']} and ends on {item['policyEndDate']}."

                return elicit_intent(intent_request, session_attributes, 
                    f'Thank you for confirming your username and PIN, {username}. {message}'
                )

            except Exception as e:
                print(f"Error querying DynamoDB: {e}")
                return e


def validate_home_insurance(intent_request, session_id, home_coverage, home_type, property_value, year_built, square_footage, home_security_system):
    """
    Validates slot values specific to Home insurance.
    """
    if home_coverage is not None:
        home_coverage_list = ['structure', 'contents', 'liability', 'all']
        if not isvalid_slot_value(home_coverage, home_coverage_list):
            prompt = "The user was asked to specify the type of home coverage [Structure, Contents, Liability, All] as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nPlease specify the type of home coverage [Structure, Contents, Liability, All]."
            return build_validation_result(False, 'HomeCoverage', reply)
    else:
        return build_validation_result(
            False,
            'HomeCoverage',
            'Please specify the type of home coverage [Structure, Contents, Liability, All].'
        )   

    if home_type is not None:
        home_type_list = ['single-family', 'multi-family', 'condo', 'townhouse']
        if not isvalid_slot_value(home_type, home_type_list):
            prompt = "The user was asked to specify the type of home [Single-Family, Multi-Family, Condo, Townhouse] as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nPlease specify the type of home [Single-Family, Multi-Family, Condo, Townhouse]."
            return build_validation_result(False, 'HomeType', reply)
    else:
        return build_validation_result(
            False,
            'HomeType',
            'Please specify the type of home [Single-Family, Multi-Family, Condo, Townhouse].'
        )

    if property_value is not None:
        if not isvalid_number(property_value):
            prompt = "The user was just asked to provide their property value as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the estimated value of your home?"
            return build_validation_result(False, 'PropertyValue', reply)
    else:
        return build_validation_result(
            False,
            'PropertyValue',
            'What is the estimated value of your home?'
        )

    if year_built is not None:
        if not isvalid_date(year_built):
            prompt = "The user was just asked to provide the year their home was built as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhich year was your home built?"
            return build_validation_result(False, 'YearBuilt', reply)
    else:
        return build_validation_result(
            False,
            'YearBuilt',
            'Which year was your home built?'
        )

    if square_footage is not None:
        if not isvalid_number(square_footage):
            prompt = "The user was just asked to provide the square footage of their home as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the square footage of your home?"
            return build_validation_result(False, 'SquareFootage', reply)
    else:
        return build_validation_result(
            False,
            'SquareFootage',
            'What is the square footage of your home?'
        )

    if home_security_system is not None:
        security_system_list = ['yes', 'no']
        if not isvalid_slot_value(home_security_system, security_system_list):
            prompt = "The user was asked if they have a home security system [Yes, No] as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nDo you have a home security system [Yes, No]?"
            return build_validation_result(False, 'HomeSecuritySystem', reply)
    else:
        return build_validation_result(
            False,
            'HomeSecuritySystem',
            'Do you have a home security system [Yes, No]?'
        )

    return {'isValid': True}

def validate_auto_insurance(intent_request, session_id, auto_coverage, age_of_insured, property_value, auto_year, annual_mileage, parking_location, previous_claims):
    """
    Validates slot values specific to Auto insurance.
    """
    print(f"validate_auto_insurance auto_coverage: {auto_coverage}")
    if auto_coverage is not None:
        print(f"auto_coverage: {auto_coverage}")
        auto_coverage_list = ['liability', 'collision', 'comprehensive', 'all']
        if not isvalid_slot_value(auto_coverage, auto_coverage_list):
            prompt = "The user was asked to specify the type of auto coverage [Liability, Collision, Comprehensive, All] as part of a home insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nPlease specify the type of auto coverage [Liability, Collision, Comprehensive, All]."
            return build_validation_result(False, 'AutoCoverage', reply)
    else:
        print("ELSE")
        return build_validation_result(
            False,
            'AutoCoverage',
            'Please specify the type of auto coverage [Liability, Collision, Comprehensive, All].'
        )   

    if age_of_insured is not None:
        if not isvalid_number(age_of_insured):
            prompt = "The user was just asked for the age of the insured on an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the age of the insured?"
            return build_validation_result(False, 'AgeOfInsured', reply)
    else:
        return build_validation_result(
            False,
            'AgeOfInsured',
            'What is the age of the insured?'
        )

    if property_value is not None:
        if not isvalid_number(property_value):
            prompt = "The user was just asked for their car value as part of an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the estimated value of your car?"
            return build_validation_result(False, 'PropertyValue', reply)
    else:
        return build_validation_result(
            False,
            'PropertyValue',
            'What is the estimated value of your car?'
        )

    if auto_year is not None:
        if not isvalid_date(auto_year):
            prompt = "The user was just asked which year their vehicle was built as part of an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhich year was your vehicle built?"
            return build_validation_result(False, 'AutoYear', reply)        
    else:
        return build_validation_result(
            False,
            'AutoYear',
            'Which year was your vehicle built?'
        )

    if annual_mileage is not None:
        if not isvalid_number(annual_mileage):
            prompt = "The user was just asked for the estimated annual mileage as part of an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the estimated annual mileage of your car?"
            return build_validation_result(False, 'AnnualMileage', reply)
    else:
        return build_validation_result(
            False,
            'AnnualMileage',
            'What is the estimated annual mileage of your car?'
        )

    if parking_location is not None:
        parking_location_list = ['garage', 'street', 'driveway']
        if not isvalid_slot_value(parking_location, parking_location_list):
            prompt = "The user was just asked where their car is usually parked (Garage, Street, Driveway) as part of an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhere is your car usually parked [Garage, Street, Driveway]?"
            return build_validation_result(False, 'ParkingLocation', reply)
    else:
        return build_validation_result(
            False,
            'ParkingLocation',
            'Where is your car usually parked [Garage, Street, Driveway]?'
        )

    if previous_claims is not None:
        previous_claims_list = ['yes', 'no']
        if not isvalid_slot_value(previous_claims, previous_claims_list):
            prompt = "The user was just asked if they have filed any auto insurance claims in the last three years ['Yes', 'No'] as part of an auto insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nHave you filed any auto insurance claims in the past three years [Yes/No]?"
            return build_validation_result(False, 'PreviousClaims', reply)
    else:
        return build_validation_result(
            False,
            'PreviousClaims',
            'Have you filed any auto insurance claims in the past three years [Yes/No]?'
        )

    return {'isValid': True}

def validate_life_insurance(intent_request, session_id, life_policy_type, age_of_insured, annual_income):
    """
    Validates slot values specific to Life insurance.
    """
    if life_policy_type is not None:
        life_policy_type_list = ['term', 'whole', 'universal', 'variable']
        if not isvalid_slot_value(life_policy_type, life_policy_type_list):
            prompt = "The user was asked to specify the type of life insurance policy [Term, Whole, Universal] and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nPlease specify the type of life insurance policy [Term, Whole, Universal]."
            return build_validation_result(False, 'LifePolicyType', reply)

    else:
        return build_validation_result(
            False,
            'LifePolicyType',
            'Please specify the type of life insurance policy [Term, Whole, Universal, Variable].'
        )

    if age_of_insured is not None:
        if not isvalid_number(age_of_insured):
            prompt = "The user was just asked for the age of the insured as part of a life insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the age of the insured?"
            return build_validation_result(False, 'AgeOfInsured', reply)
    else:
        return build_validation_result(
            False,
            'AgeOfInsured',
            'What is the age of the insured?'
        )

    if annual_income is not None:
        if not isvalid_number(annual_income):
            prompt = "The user was asked for the insured's annual income as part of a life insurance quote request and this was their response: " + intent_request['inputTranscript']
            message = invoke_agent(prompt, session_id)
            reply = message + " \n\nWhat is the insured's annual income?"
            return build_validation_result(False, 'AnnualIncome', reply)               
    else:
        return build_validation_result(
            False,
            'AnnualIncome',
            "What is the insured's annual income?"
        )

    return {'isValid': True}

def validate_insurance_quote(intent_request, username, policy_type, policy_start_date, slots):
    """
    Elicits and validates slot values provided by the user for insurance quote generation.
    """
    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    session_id = intent_request['sessionId']

    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
    else:
        try:
            session_username = intent_request['sessionState']['sessionAttributes']['UserName']
            build_slot(intent_request, 'UserName', session_username)
        except KeyError:
            return build_validation_result(
                False,
                'UserName',
                'You have been logged out. Please start a new session.'
            )

    if policy_type is not None:
        if policy_type == 'Home':
            # Home slot values
            home_coverage = try_ex(slots['HomeCoverage'])
            home_type = try_ex(slots['HomeType'])
            property_value = try_ex(slots['PropertyValue'])
            year_built = try_ex(slots['YearBuilt'])
            square_footage = try_ex(slots['SquareFootage'])
            home_security_system = try_ex(slots['HomeSecuritySystem'])

            validation_results = validate_home_insurance(intent_request, session_id, home_coverage, home_type, property_value, year_built, square_footage, home_security_system)
            if not validation_results['isValid']:
                return validation_results

        elif policy_type == 'Auto':
            # Auto slot values
            auto_coverage = try_ex(slots['AutoCoverage'])
            age_of_insured = try_ex(slots['AgeOfInsured'])
            property_value = try_ex(slots['PropertyValue'])
            auto_year = try_ex(slots['AutoYear'])
            annual_mileage = try_ex(slots['AnnualMileage'])
            parking_location = try_ex(slots['ParkingLocation'])
            previous_claims = try_ex(slots['PreviousClaims'])

            validation_results = validate_auto_insurance(intent_request, session_id, auto_coverage, age_of_insured, property_value, auto_year, annual_mileage, parking_location, previous_claims)
            if not validation_results['isValid']:
                return validation_results

        elif policy_type == 'Life':
            # Life slot values
            life_policy_type = try_ex(slots['LifePolicyType'])
            age_of_insured = try_ex(slots['AgeOfInsured'])
            annual_income = try_ex(slots['AnnualIncome'])

            validation_results = validate_life_insurance(intent_request, session_id, life_policy_type, age_of_insured, annual_income)
            if not validation_results['isValid']:
                return validation_results
    else:
        return build_validation_result(
            False,
            'PolicyType',
            'Which type of insurance policy do you need [Home, Auto, Life]?'
        )

    if policy_start_date is None:
        return build_validation_result(
            False,
            'PolicyStartDate',
            'When would you like the policy to start?'
        )

    return {'isValid': True}


def generate_insurance_quote(intent_request):
    """
    Performs dialog management and fulfillment for completing an insurance quote request.
    """
    slots = intent_request['sessionState']['intent']['slots']
    
    # Common slots regardless of 'policy_type'
    username = try_ex(slots['UserName'])
    policy_type = try_ex(slots['PolicyType'])
    policy_start_date = try_ex(slots['PolicyStartDate'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    if intent_request['invocationSource'] == 'DialogCodeHook':    
        # Validate any slots which have been specified. If any are invalid, re-elicit for their value
        input_transcript = intent_request['inputTranscript']
        policy_type_list = ['Home', 'Auto', 'Life']

        if input_transcript in policy_type_list:
            policy_type = input_transcript
            policy_type_slot = {
                "shape": "Scalar",
                "value": {
                    "originalValue": policy_type,
                    "interpretedValue": policy_type,
                    "resolvedValues": []
                }
            }
            slots['PolicyType'] = policy_type_slot
        
        validation_result = validate_insurance_quote(intent_request, username, policy_type, policy_start_date, slots)

        if not validation_result['isValid']:
            slots = intent_request['sessionState']['intent']['slots']
            slots[validation_result['violatedSlot']] = None

            return elicit_slot(
                session_attributes,
                active_contexts,
                intent_request['sessionState']['intent'],
                validation_result['violatedSlot'],
                validation_result['message']
            )

    if username and policy_type:

        # Determine which PDF to fill out based on coverage type
        pdf_template = ''
        fields_to_update = {}

        # Determine if the intent and current slot settings have been denied
        if confirmation_status == 'Denied' or confirmation_status == 'None':
            return delegate(session_attributes, active_contexts, intent, 'How else can I help you?')

        if confirmation_status == 'Confirmed':
            intent['confirmationState']="Confirmed"
            intent['state']="Fulfilled"

        # Based on policy_type, set the appropriate PDF template
        if policy_type == 'Home':
            pdf_template = 'home_insurance_template.pdf'
        elif policy_type == 'Auto':
            pdf_template = 'auto_insurance_template.pdf'
        elif policy_type == 'Life':
            pdf_template = 'life_insurance_template.pdf'

        # PDF generation and S3 upload logic
        s3_client.download_file(s3_artifact_bucket, f'agent/assets/{pdf_template}', f'/tmp/{pdf_template}')

        reader = pdfrw.PdfReader(f'/tmp/{pdf_template}')
        acroform = reader.Root.AcroForm

        # Get the fields from the PDF
        fields = reader.Root.AcroForm.Fields

        # Extract and print field names
        field_names = [field['/T'][1:-1] for field in fields if '/T' in field]

        # Loop through the slots to update fields and create fields_to_update dict
        for slot_name, slot_value in slots.items():
            field_name = slot_name.replace('_', ' ')  # Adjust field naming if necessary
            if field_name and slot_value:
                fields_to_update[field_name] = slot_value['value']['interpretedValue']

        # Update PDF fields
        if acroform is not None and '/Fields' in acroform:
            fields = acroform['/Fields']
            for field in fields:
                field_name = field['/T'][1:-1]  # Extract field name without '/'
                if field_name in fields_to_update:
                    field.update(pdfrw.PdfDict(V=fields_to_update[field_name]))

        writer = pdfrw.PdfWriter()
        writer.addpage(reader.pages[0])  # Assuming you are updating the first page

        completed_pdf_path = f'/tmp/{pdf_template.replace(".pdf", "-completed.pdf")}'
        with open(completed_pdf_path, 'wb') as output_stream:
            writer.write(output_stream)
            
        s3_client.upload_file(completed_pdf_path, s3_artifact_bucket, f'agent/assets/{pdf_template.replace(".pdf", "-completed.pdf")}')

        # Create insurance quote doc in S3
        URLs = []
        URLs.append(create_presigned_url(s3_artifact_bucket, f'agent/assets/{pdf_template.replace(".pdf", "-completed.pdf")}', 3600))
        insurance_quote_link = f'Your insurance quote request is ready! Please follow the link for details: {URLs[0]}'

        # Write insurance quote request data to DynamoDB
        quote_request = {}

        # Loop through the slots to add items to quote_request dict
        for slot_name, slot_value in slots.items():
            if slot_value:
                quote_request[slot_name] = slot_value['value']['interpretedValue']

        # Convert the JSON document to a string
        quote_request_string = json.dumps(quote_request)

        # Write the JSON document to DynamoDB
        insurance_quote_requests_table = dynamodb.Table(insurance_quote_requests_table_name)

        response = insurance_quote_requests_table.put_item(
            Item={
                'UserName': username,
                'RequestTimestamp': int(time.time()),
                'quoteRequest': quote_request_string
            }
        )

        print("Insurance Quote Request Submitted Successfully")

        return elicit_intent(
            intent_request,
            session_attributes,
            insurance_quote_link
        )


# DEV BREAK


def loan_calculator(intent_request):
    """
    Performs dialog management and fulfillment for calculating loan details.
    This is an empty function framework intended for the user to develope their own intent fulfillment functions.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}

    # def elicit_intent(intent_request, session_attributes, message)
    return elicit_intent(
        intent_request,
        session_attributes,
        'This is where you would implement LoanCalculator intent fulfillment.'
    )

def invoke_agent(prompt, session_id):
    """
    Invokes Amazon Bedrock-powered LangChain agent with 'prompt' input.
    """
    chat = Chat({'Human': prompt}, session_id)
    llm = Bedrock(client=bedrock_client, model_id="anthropic.claude-v2:1", region_name=os.environ['AWS_REGION']) # anthropic.claude-instant-v1 / anthropic.claude-3-sonnet-20240229-v1:0
    llm.model_kwargs = {'max_tokens_to_sample': 350}
    lex_agent = InsuranceAgent(llm, chat.memory)
    
    message = lex_agent.run(input=prompt)

    # Summarize response and save in memory
    formatted_prompt = "\n\nHuman: " + "Summarize the following within 50 words: " + message + " \n\nAssistant:"
    conversation = ConversationChain(llm=llm)
    ai_response_recap = conversation.predict(input=formatted_prompt)
    chat.set_memory({'Assistant': ai_response_recap}, session_id)

    return message

def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (e.g., FallbackIntent).
    Sends user utterance to the 'invoke_agent' method call.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    session_id = intent_request['sessionId']
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        prompt = intent_request['inputTranscript']
        output = invoke_agent(prompt, session_id)
        print(f"Insurance Agent response: {output}")

    return elicit_intent(intent_request, session_attributes, output)

# --- Intents ---

def dispatch(intent_request):
    """
    Routes the incoming request based on intent.
    """
    slots = intent_request['sessionState']['intent']['slots']
    username = slots['UserName'] if 'UserName' in slots else None
    intent_name = intent_request['sessionState']['intent']['name']

    if intent_name == 'VerifyIdentity':
        return verify_identity(intent_request)
    elif intent_name == 'InsuranceQuoteRequest':
        return generate_insurance_quote(intent_request)
    elif intent_name == 'LoanCalculator':
        return loan_calculator(intent_request)
    else:
        return genai_intent(intent_request)

    raise Exception('Intent with name ' + intent_name + ' not supported')
        
# --- Main handler ---

def handler(event, context):
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """
    os.environ['TZ'] = 'America/New_York'
    time.tzset()

    return dispatch(event)