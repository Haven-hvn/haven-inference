import os
import base64
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, VERSION as PYDANTIC_VERSION
from typing import List, Dict, Union, Optional
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llava15ChatHandler

MODEL_PATH = os.getenv("MODEL_PATH", "/models/smolvlm.f16.gguf")
MMPROJ_PATH = os.getenv("MMPROJ_PATH", "/models/mmproj-smolvlm.f16.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
N_CTX = int(os.getenv("N_CTX", 2048))
MODEL_ID = os.getenv("MODEL_ID", "smolvlm-v1.8b-gguf")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_dump(model_instance):
    if PYDANTIC_VERSION.startswith("1."):
        return model_instance.dict()
    else:
        return model_instance.model_dump(exclude_unset=True)

logger.info(f"+++ Starting Model Loading +++")
logger.info(f"Attempting to load Llama model from: {MODEL_PATH}")
logger.info(f"Attempting to load mmproj from: {MMPROJ_PATH}")
logger.info(f"Using N_GPU_LAYERS: {N_GPU_LAYERS}")
logger.info(f"Using N_CTX: {N_CTX}")

llm = None

try:
    model_dir = "/models"
    if not os.path.exists(MODEL_PATH):
       file_list = os.listdir(model_dir) if os.path.exists(model_dir) else 'Not Found or Empty'
       raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Contents of {model_dir}: {file_list}")
    if not os.path.exists(MMPROJ_PATH):
       file_list = os.listdir(model_dir) if os.path.exists(model_dir) else 'Not Found or Empty'
       raise FileNotFoundError(f"MMProj file not found at {MMPROJ_PATH}. Contents of {model_dir}: {file_list}")

    chat_handler = Llava15ChatHandler(clip_model_path=MMPROJ_PATH, verbose=True)

    llm = Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=True,
        verbose=True
    )
    logger.info(f"Model '{MODEL_ID}' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    llm = None

app = FastAPI(
    title="SmolVLM GGUF API",
    description="OpenAI-compatible API for SmolVLM GGUF model using llama-cpp-python",
    version="1.0.1"
)

def is_model_ready():
    return llm is not None

class ChatMessageContentPartText(BaseModel):
    type: str = "text"
    text: str

class ImageUrl(BaseModel):
    url: str

class ChatMessageContentPartImageURL(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[ChatMessageContentPartText, ChatMessageContentPartImageURL]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "cloud-deployment"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

@app.get("/v1/models", response_model=ModelList, summary="List Available Models")
async def list_models():
    if not is_model_ready():
         logger.warning("`/v1/models` called but model is not ready.")
         return ModelList(data=[])
    model_card = ModelCard(id=MODEL_ID)
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", summary="Handle VLM Chat Completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if not is_model_ready():
        logger.error("Chat completion request received, but model is not ready.")
        raise HTTPException(status_code=503, detail="Model is not ready. Please try again shortly.")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming generation is not supported.")

    if request.model != MODEL_ID:
         logger.warning(f"Request model '{request.model}' does not match loaded model '{MODEL_ID}'. Processing anyway.")

    formatted_messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            formatted_messages.append({"role": msg.role, "content": msg.content})
        else:
            content_parts = []
            for part in msg.content:
                 content_parts.append(get_model_dump(part))
            formatted_messages.append({"role": msg.role, "content": content_parts})

    try:
        logger.info(f"Generating completion for model: {request.model}")

        completion = llm.create_chat_completion(
            messages=formatted_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        logger.info("Successfully generated completion.")

        return JSONResponse(content=completion)

    except ValueError as e:
        logger.error(f"Value error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid request data or model processing error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during inference: {e}")

@app.get("/", summary="API Root/Health Check")
async def root():
    if not is_model_ready():
        return {"status": "loading","message": f"SmolVLM API ({MODEL_ID}) is initializing..."}
    return {"status": "ready", "message": f"SmolVLM API ({MODEL_ID}) is running"}

@app.get("/health", summary="Health Check Endpoint")
async def health_check():
    return {"status": "healthy", "model_ready": is_model_ready()}
