import os
import uuid
from typing import Dict, List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from azure.identity.aio import DefaultAzureCredential
from azure.ai.inference.models import UserMessage, SystemMessage, AssistantMessage
from azure.ai.projects.aio import AIProjectClient

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful AI assistant built using Azure AI Foundry. "
    "Your purpose is to provide clear, concise, and accurate information to users. "
    "Respond in a friendly and professional manner, admit when you don't know something, "
    "keep responses brief but informative, use markdown formatting where helpful, "
    "and provide complete, working examples when code is requested."
)

# Dictionary to store conversation histories by session id
conversations: Dict[str, List[Union[UserMessage, SystemMessage, AssistantMessage]]] = {}

app = FastAPI()
templates = Jinja2Templates(directory="templates")

project_connection_string = os.environ["PROJECT_CONNECTION_STRING"]
model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]

credential = DefaultAzureCredential()
project_client = AIProjectClient.from_connection_string(
    credential=credential, conn_str=project_connection_string
)


def get_or_create_session(request: Request) -> tuple[str, bool]:
    """Check if a session ID exists in the request cookies. If it does, return it; otherwise, create a new session ID."""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in conversations:
        return session_id, False

    # Create a new session
    new_session_id = str(uuid.uuid4())
    conversations[new_session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    return new_session_id, True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, response: Response) -> HTMLResponse:
    session_id, new_session = get_or_create_session(request)
    if new_session:
        print("Top New session created:", session_id)
        response.set_cookie(key="session_id", value=session_id)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat")
async def chat_stream(request: Request, prompt: str) -> StreamingResponse:
    session_id, new_session = get_or_create_session(request)
    # Append the user's message to the conversation
    conversations[session_id].append(UserMessage(content=prompt))

    async def event_generator():
        response_chunks = []
        # Get the chat completions client as an asynchronous context manager
        async with await (
            project_client.inference.get_chat_completions_client()
        ) as client:
            chat_response = await client.complete(
                model=model_deployment_name,
                messages=conversations[session_id],
                stream=True,
            )
            async for chunk in chat_response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    response_chunks.append(delta.content)
                    yield f"data: {delta.content}\n\n"

        if response_chunks:
            conversations[session_id].append(
                AssistantMessage(content="".join(response_chunks))
            )
        yield "data: [DONE]\n\n"

    stream_response = StreamingResponse(
        event_generator(), media_type="text/event-stream"
    )
    if new_session:
        stream_response.set_cookie(key="session_id", value=session_id)
    return stream_response
