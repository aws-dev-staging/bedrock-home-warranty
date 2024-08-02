import os
import json
import boto3
from langchain.agents.tools import Tool

bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name=os.environ['AWS_REGION'])

class Tools:

    def __init__(self) -> None:
        print("Initializing Tools")
        self.tools = [
            Tool(
                name="HomeWarranty",
                func=self.bedrock_query_knowledge_base,
                description="Use this tool to answer questions about First American Home Warranty.",
            )
        ]

    def bedrock_query_knowledge_base(self, query):
        print(f"Knowledge Base query: {query}")

        prompt_template = """\n\nHuman: You will be acting as a helpful customer service representative named Ava (short for Amazon Virtual Assistant) working for AnyCompany. Provide a summarized answer using only 1 or 2 sentences. 
        Here is the relevant information in numbered order from our knowledge base: $search_results$
        Current time: $current_time$
        User query: $query$\n\nAssistant: """

        model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        kb_id = "D3IJJEQWJ5"
        filter_attribute = None
        session_id = None

        payload = {
            "input": {
                "text": query
            },
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": kb_id,
                    "modelArn": model_arn,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5,
                        }
                    }
                }
            }
        }

        if filter_attribute is not None:
            print(f"filter_attribute: {filter_attribute}")
            payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["retrievalConfiguration"] = {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "filter": {
                        "equals": {
                            "key": "exposure",
                            "value": filter_attribute
                        }
                    }
                }
            }

        if session_id is not None:
            print(f"session_id: {session_id}")
            payload["sessionId"] = session_id

        try:
            response = bedrock_agent_runtime_client.retrieve_and_generate(**payload)
            print(f"BEDROCK RESPONSE: {response}")
            retrieval_results = response.get("retrievalResults", [])
            generated_text = ""

            for citation in response.get('citations', []):
                if 'generatedResponsePart' in citation:
                    generated_text += citation['generatedResponsePart']['textResponsePart']['text'] + " "

            generated_text = generated_text.strip()

            # Extract and append URLs from retrievedReferences
            sources = []
            for citation in response.get('citations', []):
                for ref in citation.get('retrievedReferences', []):
                    location = ref.get('location', {})
                    url = None
                    if 'webLocation' in location:
                        url = location['webLocation'].get('url')
                    elif 's3Location' in location:
                        url = location['s3Location'].get('uri')
                    elif 'confluenceLocation' in location:
                        url = location['confluenceLocation'].get('url')
                    elif 'salesforceLocation' in location:
                        url = location['salesforceLocation'].get('url')
                    elif 'sharePointLocation' in location:
                        url = location['sharePointLocation'].get('url')

                    if url:
                        sources.append(url)

            if sources:
                sources_text = "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
                generated_text += sources_text

            if generated_text:
                print(f"Knowledge Base response: {generated_text}\n")
                return generated_text
            else:
                return "No relevant information found in the knowledge base."

        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return f"Error querying knowledge base: {e}"


# Pass the initialized retriever and llm to the Tools class constructor
tools = Tools().tools
