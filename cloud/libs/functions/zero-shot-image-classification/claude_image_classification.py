import json
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import os
import re

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize clients
REGION_NAME = os.environ.get('S3_BUCKET_REGION', 'eu-west-2')

bedrock_runtime = boto3.client('bedrock-runtime', REGION_NAME)
s3 = boto3.client('s3')

iot_data_client = boto3.client('iot-data')


def invoke_claude_classifier(event):
    """
    Classifies images using Claude, given an S3 event trigger
    
    Args:
        event: AWS event object containing S3 details
        
    Returns:
        tuple: (classification_dict, item_dict) containing classification and confidence scores
    """
    # Extract event details
    BUCKET_NAME = event["detail"]["bucket"]["name"]
    KEY = event["detail"]["object"]["key"]
    
    # Get image data from S3
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=KEY)
        image_data = response['Body'].read()
    except ClientError as e:
        print(f"Error reading from S3: {e}")
        raise

    prompt = '''
You are an advanced image analysis system specialized in waste categorization. Your task is to analyze images of waste items and provide accurate classification and identification information. This information is crucial for efficient waste management and recycling processes.

Your objective is to provide two key pieces of information about the waste item shown in the image:

1. The waste category (choose exactly one: organic, landfill, or recycle)
2. A brief label (1-3 words) identifying the main item shown

Before providing your final output, wrap your analysis inside <waste_analysis> tags. In your analysis, follow these steps:

1. Identify the main item in the image and describe its key features.
2. List the materials that compose the item (e.g., plastic, metal, food waste).
3. Consider the following for each potential category:
   a. Organic:
      - Pros: List any organic or biodegradable components.
      - Cons: List any non-organic or non-biodegradable components.
   b. Recycle:
      - Pros: List any recyclable materials or components.
      - Cons: List any non-recyclable materials or contaminants.
   c. Landfill:
      - Pros: List any reasons why the item might not fit in other categories.
      - Cons: List any reasons why the item could potentially belong in other categories.
4. Determine if the item contains any organic matter that can decompose or spoil.
5. Assess if the item has recyclable components or raw materials.
6. Consider the waste categorization rules:
   a. If there are organic materials that can decompose or spoil, classify as organic.
   b. If the item has recyclable components, classify as recycle.
   c. If the item doesn't fit into organic or recycle categories, classify as landfill.
7. Prioritize categorization in this order: organic > recycle > landfill.
8. Estimate your confidence in both the category and item identification.

After your analysis, provide your final output in the following format:

<final_output>
Category: [waste category]
Category_confidence: [confidence level as a decimal]
Item: [1-3 word item description]
Item_confidence: [confidence level as a decimal]
</final_output>

Ensure that:
- The category is exactly one of: organic, landfill, or recycle
- The item label is 1-3 words
- Confidence scores are decimal numbers between 0 and 1, with two decimal places

Here's a generic example of how your output should be formatted:

Category: landfill
Category_confidence: 0.87
Item: a broken spoon
Item_confidence: 0.93

Remember to replace the placeholders with your actual analysis results.
'''

    # Single message requesting both classifications
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": image_data}
                    }
                },
                {
                    "text": prompt
                }
            ]
        }
    ]
    
    try:
        # Get both classifications in one call
        response = get_claude_response(messages)
        print(f"DEBUG: {response}")
        classification_dict, item_dict, reason = parse_claude_response(response)
        
        return classification_dict, item_dict, reason
        
    except Exception as e:
        print(f"Error during Claude classification: {e}")
        raise

def parse_claude_response(response):
    """
    Parse Claude's response into two dictionaries with Name and Score, plus analysis text
    
    Returns:
        tuple: (classification_dict, item_dict, analysis_text)
    """
    try:
        # Extract the waste analysis section
        analysis_match = re.search(r'<waste_analysis>(.*?)</waste_analysis>', response, re.DOTALL)
        analysis_text = analysis_match.group(1).strip() if analysis_match else ""
        
        # Extract the final output section
        final_output_match = re.search(r'<final_output>(.*?)</final_output>', response, re.DOTALL)
        if not final_output_match:
            print("No final_output section found")
            raise ValueError("Missing final_output section")
            
        final_output = final_output_match.group(1).strip()
        
        # Parse the final output section
        response_dict = {}
        for line in final_output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                response_dict[key.strip()] = value.strip()
        
        if not all(k in response_dict for k in ['Category', 'Category_confidence', 'Item', 'Item_confidence']):
            raise ValueError("Missing required fields in output")
            
        classification = {
            "Name": response_dict["Category"].lower(),
            "Score": float(response_dict["Category_confidence"])
        }
        
        item = {
            "Name": response_dict["Item"].lower(),
            "Score": float(response_dict["Item_confidence"])
        }
        
        return classification, item, analysis_text
        
    except Exception as e:
        print(f"Error parsing Claude response: {e}")
        print(f"Response was: {response}")
        # Return default responses if parsing fails
        default_dict = {"Name": "unknown", "Score": 0.0}
        return default_dict, default_dict, ""


def get_claude_response(messages):
    """
    Helper function to get response from Claude
    """

    inference_config = {
        "temperature": 0.0,
        "maxTokens": 1000,
        "topP": 1,
    }
    
    api_params = {
        "modelId": MODEL_ID,
        "messages": messages,
        "inferenceConfig": inference_config,
        "system": [{"text": "You are a waste classification expert. Always provide your response in the exact format requested, with categories and confidence scores."}]
    }
    
    try:
        response = bedrock_runtime.converse(**api_params)
        return response['output']['message']['content'][0]['text'].strip()
    except ClientError as err:
        print(f"Error calling Claude: {err.response['Error']['Message']}")
        raise


def updateShadowTopic(result):
    try:
        payload = {
            "state": {
                "desired": {
                    "classification": result
                }
            }
        }
        
        payload = json.dumps(payload).encode('utf-8')
        
        # Attempt to publish to IoT Topic
        try:
            iot_data_client.publish(
                topic='$aws/things/DemoWasteBin/shadow/update',
                qos=0,
                payload=payload
            )
            print(f"Successfully published classification {result} to IoT shadow")
            return True
            
        except Exception as iot_error:
            print(f"Error publishing to IoT shadow: {str(iot_error)}")
            # Optionally attempt retry logic here if needed
            return False
            
    except Exception as e:
        print(f"Error preparing payload: {str(e)}")
        return False    


def lambda_handler(event, context):

    classification, item_label, reason = invoke_claude_classifier(event)

    # print("classification:", classification)
    # print("item_label:", item_label)

    event["classification"] = classification["Name"]
    event["Score"] = classification["Score"]
    event["Item"] = item_label["Name"]
    event["Item_Score"] = item_label["Score"]
    event["Reason"] = reason

    updateShadowTopic(event["classification"])


    return event
