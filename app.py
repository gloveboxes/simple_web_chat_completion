import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from azure.identity.aio import DefaultAzureCredential
from azure.ai.inference.models import UserMessage
from azure.ai.projects.aio import AIProjectClient

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

project_connection_string = os.environ["PROJECT_CONNECTION_STRING"]
model_deployment_name = os.environ["MODEL_DEPLOYMENT_NAME"]

credential = DefaultAzureCredential()
project_client = AIProjectClient.from_connection_string(
    credential=credential,
    conn_str=project_connection_string
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat_stream(request: Request, prompt: str):
    async def event_generator():
        async with await project_client.inference.get_chat_completions_client() as client:
            response = await client.complete(
                model=model_deployment_name,
                messages=[UserMessage(content=prompt)],
                stream=True,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
