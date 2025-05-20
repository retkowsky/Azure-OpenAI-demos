import json
import os
import traceback

from azure.identity import DefaultAzureCredential
from dotenv import find_dotenv, load_dotenv
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from semantic_kernel.contents import ChatHistory


class SOPAgent:
    _agentName = "SOPAgent"

    async def execute(self, video_steps: list[str]) -> str:
        load_dotenv("cu/cu.env")
        
        AGENT_INSTRUCTIONS_PATH = os.getenv("AGENT_INSTRUCTIONS_PATH")
        AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")
        AZURE_AI_AGENT_AGENT_ID = os.getenv("AZURE_AI_AGENT_AGENT_ID")
        AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")

        analysis_text = ""

        # Create an Azure AI agent client
        client = AzureAIAgent.create_client(
            credential=DefaultAzureCredential(),
            conn_str=AZURE_AI_AGENT_PROJECT_CONNECTION_STRING)

        agent_definition = None

        if not AZURE_AI_AGENT_AGENT_ID:
            with open(AGENT_INSTRUCTIONS_PATH, 'r') as file:
                instructions = file.read()
                agent_definition = await client.agents.create_agent(
                    name="SOPAgentSample",
                    description="SOP Agent to produce a Standard Operating Procedure from a video file.",
                    instructions=instructions,
                    model=AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME,
                )
        else:
            # Retrieve an existing agent
            agent_definition = await client.agents.get_agent(
                agent_id=AZURE_AI_AGENT_AGENT_ID)

        # Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )

        # Create a thread for the agent
        thread: AzureAIAgentThread = AzureAIAgentThread(messages=ChatHistory(),
                                                        client=client)

        try:
            analysis_text = await agent.get_response(
                messages=json.dumps(video_steps), thread=thread)
            analysis_text = analysis_text.content.content           
        except Exception as e:
            stack_trace = traceback.format_exc()
        finally:
            await thread.delete() if thread else None
            await client.close() if client else None

        return analysis_text

