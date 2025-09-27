# NVIDIA API Integration for FLOAT CHAT
# This router now routes all Ollama-compatible requests through NVIDIA API endpoints
# Provides seamless integration with NVIDIA's cloud-based models

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime

from typing import Optional, Union, List, Dict, Any
from urllib.parse import urlparse
import aiohttp
from aiocache import cached
import requests
from urllib.parse import quote
from openai import OpenAI

from open_webui.models.chats import Chats
from open_webui.models.users import UserModel

from open_webui.env import (
    ENABLE_FORWARD_USER_INFO_HEADERS,
)

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, validator
from starlette.background import BackgroundTask


from open_webui.models.models import Models
from open_webui.utils.misc import (
    calculate_sha256,
)
from open_webui.utils.payload import (
    apply_model_params_to_body_ollama,
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access


from open_webui.config import (
    UPLOAD_DIR,
)
from open_webui.env import (
    ENV,
    SRC_LOG_LEVELS,
    MODELS_CACHE_TTL,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OLLAMA"])

# NVIDIA API Configuration
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Available NVIDIA models that replace Ollama models
NVIDIA_MODELS = [
    {
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "name": "Qwen3 Coder 480B A35B Instruct",
        "size": 0,  # Cloud-based, no local size
        "digest": "nvidia-cloud-model",
        "details": {
            "parent_model": "",
            "format": "nvidia-api",
            "family": "qwen",
            "families": ["qwen"],
            "parameter_size": "480B",
            "quantization_level": "fp16"
        },
        "urls": [0],  # Default URL index
        "api_key_env": "NVIDIA_QWEN_API_KEY"
    },
    {
        "model": "moonshotai/kimi-k2-instruct-0905",
        "name": "Moonshot AI Kimi K2 Instruct 0905", 
        "size": 0,
        "digest": "nvidia-cloud-model",
        "details": {
            "parent_model": "",
            "format": "nvidia-api",
            "family": "kimi",
            "families": ["kimi"],
            "parameter_size": "unknown",
            "quantization_level": "fp16"
        },
        "urls": [0],
        "api_key_env": "NVIDIA_KIMI_API_KEY"
    },
    {
        "model": "deepseek-ai/deepseek-r1-0528",
        "name": "DeepSeek R1 0528",
        "size": 0,
        "digest": "nvidia-cloud-model", 
        "details": {
            "parent_model": "",
            "format": "nvidia-api",
            "family": "deepseek",
            "families": ["deepseek"],
            "parameter_size": "unknown",
            "quantization_level": "fp16"
        },
        "urls": [0],
        "api_key_env": "NVIDIA_DEEPSEEK_API_KEY"
    },
    {
        "model": "meta/llama-3.1-nemotron-70b-instruct",
        "name": "Llama 3.1 Nemotron 70B Instruct",
        "size": 0,
        "digest": "nvidia-cloud-model",
        "details": {
            "parent_model": "",
            "format": "nvidia-api", 
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "70B",
            "quantization_level": "fp16"
        },
        "urls": [0],
        "api_key_env": "NVIDIA_API_KEY"
    },
    {
        "model": "meta/llama-3.1-405b-instruct",
        "name": "Llama 3.1 405B Instruct",
        "size": 0,
        "digest": "nvidia-cloud-model",
        "details": {
            "parent_model": "",
            "format": "nvidia-api",
            "family": "llama", 
            "families": ["llama"],
            "parameter_size": "405B",
            "quantization_level": "fp16"
        },
        "urls": [0],
        "api_key_env": "NVIDIA_API_KEY"
    }
]


def get_nvidia_api_key(model_name: str = None):
    """Get NVIDIA API key from environment variables based on model"""
    if model_name:
        for model in NVIDIA_MODELS:
            if model["model"] == model_name:
                api_key = os.getenv(model["api_key_env"])
                if api_key:
                    return api_key
    
    # Fallback to default NVIDIA API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA API key not configured. Please set the appropriate NVIDIA API key environment variable."
        )
    return api_key


def get_nvidia_headers(model_name: str = None):
    """Get headers for NVIDIA API requests"""
    return {
        "Authorization": f"Bearer {get_nvidia_api_key(model_name)}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


##########################################
#
# Utility functions
#
##########################################


async def send_get_request(url, key=None, user: UserModel = None):
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(
                url,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {key}"} if key else {}),
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f"Connection error: {e}")
        return None


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


async def send_post_request(
    url: str,
    payload: Union[str, bytes],
    stream: bool = True,
    key: Optional[str] = None,
    content_type: Optional[str] = None,
    user: UserModel = None,
    metadata: Optional[dict] = None,
):

    r = None
    try:
        session = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        )

        r = await session.post(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                        **(
                            {"X-OpenWebUI-Chat-Id": metadata.get("chat_id")}
                            if metadata and metadata.get("chat_id")
                            else {}
                        ),
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        if r.ok is False:
            try:
                res = await r.json()
                await cleanup_response(r, session)
                if "error" in res:
                    raise HTTPException(status_code=r.status, detail=res["error"])
            except HTTPException as e:
                raise e  # Re-raise HTTPException to be handled by FastAPI
            except Exception as e:
                log.error(f"Failed to parse error response: {e}")
                raise HTTPException(
                    status_code=r.status,
                    detail=f"FLOAT CHAT: Server Connection Error",
                )

        r.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        if stream:
            response_headers = dict(r.headers)

            if content_type:
                response_headers["Content-Type"] = content_type

            return StreamingResponse(
                r.content,
                status_code=r.status,
                headers=response_headers,
                background=BackgroundTask(
                    cleanup_response, response=r, session=session
                ),
            )
        else:
            res = await r.json()
            return res

    except HTTPException as e:
        raise e  # Re-raise HTTPException to be handled by FastAPI
    except Exception as e:
        detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status if r else 500,
            detail=detail if e else "FLOAT CHAT: Server Connection Error",
        )
    finally:
        if not stream:
            await cleanup_response(r, session)


def get_api_key(idx, url, configs):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return configs.get(str(idx), configs.get(base_url, {})).get(
        "key", None
    )  # Legacy support


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.head("/")
@router.get("/")
async def get_status():
    return {"status": True}


class ConnectionVerificationForm(BaseModel):
    url: str
    key: Optional[str] = None


@router.post("/verify")
async def verify_connection(
    form_data: ConnectionVerificationForm, user=Depends(get_admin_user)
):
    url = form_data.url
    key = form_data.key

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            async with session.get(
                f"{url}/api/version",
                headers={
                    **({"Authorization": f"Bearer {key}"} if key else {}),
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                if r.status != 200:
                    detail = f"HTTP Error: {r.status}"
                    res = await r.json()

                    if "error" in res:
                        detail = f"External Error: {res['error']}"
                    raise Exception(detail)

                data = await r.json()
                return data
        except aiohttp.ClientError as e:
            log.exception(f"Client error: {str(e)}")
            raise HTTPException(
                status_code=500, detail="FLOAT CHAT: Server Connection Error"
            )
        except Exception as e:
            log.exception(f"Unexpected error: {e}")
            error_detail = f"Unexpected error: {str(e)}"
            raise HTTPException(status_code=500, detail=error_detail)


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "ENABLE_OLLAMA_API": True,  # Always enabled since we're using NVIDIA
        "OLLAMA_BASE_URLS": [NVIDIA_API_BASE_URL],  # Return NVIDIA URL for compatibility
        "OLLAMA_API_CONFIGS": {},  # Empty configs since we use environment variables
        "NVIDIA_MODELS": NVIDIA_MODELS,  # Include available NVIDIA models
    }


class OllamaConfigForm(BaseModel):
    ENABLE_OLLAMA_API: Optional[bool] = None
    OLLAMA_BASE_URLS: list[str]
    OLLAMA_API_CONFIGS: dict


@router.post("/config/update")
async def update_config(
    request: Request, form_data: OllamaConfigForm, user=Depends(get_admin_user)
):
    # For compatibility, we accept the config but don't actually change anything
    # since we're routing through NVIDIA API
    log.info("Config update request received - routing through NVIDIA API")

    return {
        "ENABLE_OLLAMA_API": True,  # Always enabled
        "OLLAMA_BASE_URLS": [NVIDIA_API_BASE_URL],
        "OLLAMA_API_CONFIGS": {},
        "NVIDIA_MODELS": NVIDIA_MODELS,
    }


def merge_ollama_models_lists(model_lists):
    merged_models = {}

    for idx, model_list in enumerate(model_lists):
        if model_list is not None:
            for model in model_list:
                id = model.get("model")
                if id is not None:
                    if id not in merged_models:
                        model["urls"] = [idx]
                        merged_models[id] = model
                    else:
                        merged_models[id]["urls"].append(idx)

    return list(merged_models.values())


@cached(
    ttl=MODELS_CACHE_TTL,
    key=lambda _, user: f"nvidia_all_models_{user.id}" if user else "nvidia_all_models",
)
async def get_all_models(request: Request, user: UserModel = None):
    log.info("get_all_models() - Using NVIDIA API")
    
    # Filter models based on available API keys
    available_models = []
    for model in NVIDIA_MODELS:
        api_key = os.getenv(model["api_key_env"])
        if api_key:  # Only include models with valid API keys
            available_models.append(model)
    
    models = {"models": available_models}
    
    # Store models in app state for compatibility
    request.app.state.OLLAMA_MODELS = {
        model["model"]: model for model in models["models"]
    }
    return models


async def get_filtered_models(models, user):
    # Filter models based on user access control
    filtered_models = []
    for model in models.get("models", []):
        model_info = Models.get_model_by_id(model["model"])
        if model_info:
            if user.id == model_info.user_id or has_access(
                user.id, type="read", access_control=model_info.access_control
            ):
                filtered_models.append(model)
    return filtered_models


@router.get("/api/tags")
@router.get("/api/tags/{url_idx}")
async def get_ollama_tags(
    request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)
):
    models = []

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
        key = get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS)

        r = None
        try:
            r = requests.request(
                method="GET",
                url=f"{url}/api/tags",
                headers={
                    **({"Authorization": f"Bearer {key}"} if key else {}),
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
            )
            r.raise_for_status()

            models = r.json()
        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"Ollama: {res['error']}"
                except Exception:
                    detail = f"Ollama: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=detail if detail else "FLOAT CHAT: Server Connection Error",
            )

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models["models"] = await get_filtered_models(models, user)

    return models


@router.get("/api/ps")
async def get_ollama_loaded_models(request: Request, user=Depends(get_admin_user)):
    """
    List models that are currently loaded into Ollama memory, and which node they are loaded on.
    """
    if request.app.state.config.ENABLE_OLLAMA_API:
        request_tasks = []
        for idx, url in enumerate(request.app.state.config.OLLAMA_BASE_URLS):
            if (str(idx) not in request.app.state.config.OLLAMA_API_CONFIGS) and (
                url not in request.app.state.config.OLLAMA_API_CONFIGS  # Legacy support
            ):
                request_tasks.append(send_get_request(f"{url}/api/ps", user=user))
            else:
                api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
                    str(idx),
                    request.app.state.config.OLLAMA_API_CONFIGS.get(
                        url, {}
                    ),  # Legacy support
                )

                enable = api_config.get("enable", True)
                key = api_config.get("key", None)

                if enable:
                    request_tasks.append(
                        send_get_request(f"{url}/api/ps", key, user=user)
                    )
                else:
                    request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, None)))

        responses = await asyncio.gather(*request_tasks)

        for idx, response in enumerate(responses):
            if response:
                url = request.app.state.config.OLLAMA_BASE_URLS[idx]
                api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
                    str(idx),
                    request.app.state.config.OLLAMA_API_CONFIGS.get(
                        url, {}
                    ),  # Legacy support
                )

                prefix_id = api_config.get("prefix_id", None)

                for model in response.get("models", []):
                    if prefix_id:
                        model["model"] = f"{prefix_id}.{model['model']}"

        models = {
            "models": merge_ollama_models_lists(
                map(
                    lambda response: response.get("models", []) if response else None,
                    responses,
                )
            )
        }
    else:
        models = {"models": []}

    return models


@router.get("/api/version")
@router.get("/api/version/{url_idx}")
async def get_ollama_versions(request: Request, url_idx: Optional[int] = None):
    # Return NVIDIA API version information
    return {
        "version": "1.0.0-nvidia",
        "provider": "nvidia",
        "api_base": NVIDIA_API_BASE_URL,
        "models_available": len([m for m in NVIDIA_MODELS if os.getenv(m["api_key_env"])])
    }


class ModelNameForm(BaseModel):
    model: Optional[str] = None
    model_config = ConfigDict(
        extra="allow",
    )


@router.post("/api/unload")
async def unload_model(
    request: Request,
    form_data: ModelNameForm,
    user=Depends(get_admin_user),
):
    form_data = form_data.model_dump(exclude_none=True)
    model_name = form_data.get("model", form_data.get("name"))

    if not model_name:
        raise HTTPException(
            status_code=400, detail="Missing name of the model to unload."
        )

    # Refresh/load models if needed, get mapping from name to URLs
    await get_all_models(request, user=user)
    models = request.app.state.OLLAMA_MODELS

    # Canonicalize model name (if not supplied with version)
    if ":" not in model_name:
        model_name = f"{model_name}:latest"

    if model_name not in models:
        raise HTTPException(
            status_code=400, detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_name)
        )
    url_indices = models[model_name]["urls"]

    # Send unload to ALL url_indices
    results = []
    errors = []
    for idx in url_indices:
        url = request.app.state.config.OLLAMA_BASE_URLS[idx]
        api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
            str(idx), request.app.state.config.OLLAMA_API_CONFIGS.get(url, {})
        )
        key = get_api_key(idx, url, request.app.state.config.OLLAMA_API_CONFIGS)

        prefix_id = api_config.get("prefix_id", None)
        if prefix_id and model_name.startswith(f"{prefix_id}."):
            model_name = model_name[len(f"{prefix_id}.") :]

        payload = {"model": model_name, "keep_alive": 0, "prompt": ""}

        try:
            res = await send_post_request(
                url=f"{url}/api/generate",
                payload=json.dumps(payload),
                stream=False,
                key=key,
                user=user,
            )
            results.append({"url_idx": idx, "success": True, "response": res})
        except Exception as e:
            log.exception(f"Failed to unload model on node {idx}: {e}")
            errors.append({"url_idx": idx, "success": False, "error": str(e)})

    if len(errors) > 0:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model on {len(errors)} nodes: {errors}",
        )

    return {"status": True}


@router.post("/api/pull")
@router.post("/api/pull/{url_idx}")
async def pull_model(
    request: Request,
    form_data: ModelNameForm,
    url_idx: int = 0,
    user=Depends(get_admin_user),
):
    form_data = form_data.model_dump(exclude_none=True)
    model_name = form_data.get("model", form_data.get("name"))
    
    log.info(f"Pull request for NVIDIA model: {model_name}")
    
    # Check if the model exists in our NVIDIA models list
    nvidia_model = None
    for model in NVIDIA_MODELS:
        if model["model"] == model_name or model["model"].endswith(model_name):
            nvidia_model = model
            break
    
    if nvidia_model:
        # Check if API key is available
        api_key = os.getenv(nvidia_model["api_key_env"])
        if api_key:
            return {
                "status": "success",
                "message": f"NVIDIA model {model_name} is ready (cloud-based)"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"API key not configured for model {model_name}. Please set {nvidia_model['api_key_env']} environment variable."
            )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"NVIDIA model {model_name} not found"
        )


class PushModelForm(BaseModel):
    model: str
    insecure: Optional[bool] = None
    stream: Optional[bool] = None


@router.delete("/api/push")
@router.delete("/api/push/{url_idx}")
async def push_model(
    request: Request,
    form_data: PushModelForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx is None:
        await get_all_models(request, user=user)
        models = request.app.state.OLLAMA_MODELS

        if form_data.model in models:
            url_idx = models[form_data.model]["urls"][0]
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
    log.debug(f"url: {url}")

    return await send_post_request(
        url=f"{url}/api/push",
        payload=form_data.model_dump_json(exclude_none=True).encode(),
        key=get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS),
        user=user,
    )


class CreateModelForm(BaseModel):
    model: Optional[str] = None
    stream: Optional[bool] = None
    path: Optional[str] = None

    model_config = ConfigDict(extra="allow")


@router.post("/api/create")
@router.post("/api/create/{url_idx}")
async def create_model(
    request: Request,
    form_data: CreateModelForm,
    url_idx: int = 0,
    user=Depends(get_admin_user),
):
    log.debug(f"form_data: {form_data}")
    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]

    return await send_post_request(
        url=f"{url}/api/create",
        payload=form_data.model_dump_json(exclude_none=True).encode(),
        key=get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS),
        user=user,
    )


class CopyModelForm(BaseModel):
    source: str
    destination: str


@router.post("/api/copy")
@router.post("/api/copy/{url_idx}")
async def copy_model(
    request: Request,
    form_data: CopyModelForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx is None:
        await get_all_models(request, user=user)
        models = request.app.state.OLLAMA_MODELS

        if form_data.source in models:
            url_idx = models[form_data.source]["urls"][0]
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.source),
            )

    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
    key = get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS)

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/copy",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        log.debug(f"r.text: {r.text}")
        return True
    except Exception as e:
        log.exception(e)

        detail = None
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    detail = f"Ollama: {res['error']}"
            except Exception:
                detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=detail if detail else "FLOAT CHAT: Server Connection Error",
        )


@router.delete("/api/delete")
@router.delete("/api/delete/{url_idx}")
async def delete_model(
    request: Request,
    form_data: ModelNameForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    form_data = form_data.model_dump(exclude_none=True)
    model_name = form_data.get("model", form_data.get("name"))

    log.info(f"Delete request for NVIDIA model: {model_name}")
    
    # NVIDIA models are cloud-based and cannot be deleted
    raise HTTPException(
        status_code=501,
        detail="Model deletion is not supported for NVIDIA cloud models"
    )


@router.post("/api/show")
async def show_model_info(
    request: Request, form_data: ModelNameForm, user=Depends(get_verified_user)
):
    form_data = form_data.model_dump(exclude_none=True)
    model_name = form_data.get("model", form_data.get("name"))

    log.info(f"Show model info request for: {model_name}")

    # Find the model in our NVIDIA models list
    nvidia_model = None
    for model in NVIDIA_MODELS:
        if model["model"] == model_name or model["model"].endswith(model_name):
            nvidia_model = model
            break

    if not nvidia_model:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model_name),
        )

    # Return model information in Ollama format
    return {
        "modelfile": f"# NVIDIA Cloud Model: {nvidia_model['name']}\nFROM {nvidia_model['model']}\n",
        "parameters": f"# Model: {nvidia_model['model']}\n# Provider: NVIDIA\n# Parameter Size: {nvidia_model['details']['parameter_size']}\n",
        "template": "{{ .Prompt }}",
        "details": nvidia_model["details"],
        "model_info": {
            "general.architecture": nvidia_model["details"]["family"],
            "general.file_type": 1,
            "general.parameter_count": nvidia_model["details"]["parameter_size"],
            "general.quantization_version": 2
        }
    }


class GenerateEmbedForm(BaseModel):
    model: str
    input: list[str] | str
    truncate: Optional[bool] = None
    options: Optional[dict] = None
    keep_alive: Optional[Union[int, str]] = None


@router.post("/api/embed")
@router.post("/api/embed/{url_idx}")
async def embed(
    request: Request,
    form_data: GenerateEmbedForm,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):
    log.info(f"generate_ollama_batch_embeddings {form_data}")

    if url_idx is None:
        await get_all_models(request, user=user)
        models = request.app.state.OLLAMA_MODELS

        model = form_data.model

        if ":" not in model:
            model = f"{model}:latest"

        if model in models:
            url_idx = random.choice(models[model]["urls"])
        else:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(form_data.model),
            )

    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
    api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
        str(url_idx),
        request.app.state.config.OLLAMA_API_CONFIGS.get(url, {}),  # Legacy support
    )
    key = get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS)

    prefix_id = api_config.get("prefix_id", None)
    if prefix_id:
        form_data.model = form_data.model.replace(f"{prefix_id}.", "")

    try:
        r = requests.request(
            method="POST",
            url=f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            data=form_data.model_dump_json(exclude_none=True).encode(),
        )
        r.raise_for_status()

        data = r.json()
        return data
    except Exception as e:
        log.exception(e)

        detail = None
        if r is not None:
            try:
                res = r.json()
                if "error" in res:
                    detail = f"Ollama: {res['error']}"
            except Exception:
                detail = f"Ollama: {e}"

        raise HTTPException(
            status_code=r.status_code if r else 500,
            detail=detail if detail else "FLOAT CHAT: Server Connection Error",
        )


class GenerateEmbeddingsForm(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = None
    keep_alive: Optional[Union[int, str]] = None


@router.post("/api/embeddings")
@router.post("/api/embeddings/{url_idx}")
async def embeddings(
    request: Request,
    form_data: GenerateEmbeddingsForm,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):
    log.info(f"generate_nvidia_embeddings {form_data}")

    # Find the model in our NVIDIA models list
    model_name = form_data.model.replace(":latest", "")
    nvidia_model = None
    for model in NVIDIA_MODELS:
        if model["model"] == model_name or model["model"].endswith(model_name):
            nvidia_model = model
            break
    
    if not nvidia_model:
        # Default to first available model if not found
        available_models = [m for m in NVIDIA_MODELS if os.getenv(m["api_key_env"])]
        if not available_models:
            raise HTTPException(
                status_code=500,
                detail="No NVIDIA models available. Please configure API keys."
            )
        nvidia_model = available_models[0]
    
    try:
        payload = {
            "model": nvidia_model["model"],
            "input": form_data.prompt
        }
        
        headers = get_nvidia_headers(nvidia_model["model"])
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{NVIDIA_API_BASE_URL}/embeddings",
                json=payload,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"NVIDIA embeddings API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"NVIDIA embeddings API error: {error_text}"
                    )
                
                result = await response.json()
                
                # Convert OpenAI format to Ollama format
                if 'data' in result and len(result['data']) > 0:
                    embedding_data = result['data'][0]
                    ollama_response = {
                        "embedding": embedding_data.get('embedding', [])
                    }
                    return ollama_response
                else:
                    return result
                
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error generating NVIDIA embeddings: {e}")
        raise HTTPException(
            status_code=500,
            detail=ERROR_MESSAGES.DEFAULT(f"Error generating embeddings: {e}")
        )


class GenerateCompletionForm(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    images: Optional[list[str]] = None
    format: Optional[Union[dict, str]] = None
    options: Optional[dict] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[list[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = None
    keep_alive: Optional[Union[int, str]] = None


@router.post("/api/generate")
@router.post("/api/generate/{url_idx}")
async def generate_completion(
    request: Request,
    form_data: GenerateCompletionForm,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):
    # Convert Ollama generate format to chat completion format for NVIDIA API
    messages = []
    if form_data.system:
        messages.append({"role": "system", "content": form_data.system})
    
    messages.append({"role": "user", "content": form_data.prompt})
    
    # Create a chat completion payload
    chat_payload = {
        "model": form_data.model,
        "messages": messages,
        "temperature": form_data.options.get("temperature", 0.7) if form_data.options else 0.7,
        "top_p": form_data.options.get("top_p", 0.8) if form_data.options else 0.8,
        "max_tokens": form_data.options.get("num_predict", 4096) if form_data.options else 4096,
        "stream": form_data.stream
    }
    
    return await generate_nvidia_chat_completion(request, chat_payload, user)


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    images: Optional[list[str]] = None

    @validator("content", pre=True)
    @classmethod
    def check_at_least_one_field(cls, field_value, values, **kwargs):
        # Raise an error if both 'content' and 'tool_calls' are None
        if field_value is None and (
            "tool_calls" not in values or values["tool_calls"] is None
        ):
            raise ValueError(
                "At least one of 'content' or 'tool_calls' must be provided"
            )

        return field_value


class GenerateChatCompletionForm(BaseModel):
    model: str
    messages: list[ChatMessage]
    format: Optional[Union[dict, str]] = None
    options: Optional[dict] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    keep_alive: Optional[Union[int, str]] = None
    tools: Optional[list[dict]] = None
    model_config = ConfigDict(
        extra="allow",
    )


async def get_ollama_url(request: Request, model: str, url_idx: Optional[int] = None):
    if url_idx is None:
        models = request.app.state.OLLAMA_MODELS
        if model not in models:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MESSAGES.MODEL_NOT_FOUND(model),
            )
        url_idx = random.choice(models[model].get("urls", []))
    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
    return url, url_idx


@router.post("/api/chat")
@router.post("/api/chat/{url_idx}")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    metadata = form_data.pop("metadata", None)
    try:
        form_data = GenerateChatCompletionForm(**form_data)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    if isinstance(form_data, BaseModel):
        payload = {**form_data.model_dump(exclude_none=True)}

    if "metadata" in payload:
        del payload["metadata"]

    model_id = payload["model"]
    model_info = Models.get_model_by_id(model_id)

    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop("system", None)

            payload = apply_model_params_to_body_ollama(params, payload)
            payload = apply_system_prompt_to_body(system, payload, metadata, user)

        # Check if user has access to the model
        if not bypass_filter and user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(
                    user.id, type="read", access_control=model_info.access_control
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    elif not bypass_filter:
        if user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model not found",
            )

    # Route through NVIDIA API instead of Ollama
    return await generate_nvidia_chat_completion(request, payload, user, metadata)


async def generate_nvidia_chat_completion(request: Request, payload: dict, user: UserModel, metadata: dict = None):
    """Generate chat completion using NVIDIA API with OpenAI client - EXACT implementation as requested"""
    try:
        log.info(f"NVIDIA chat completion request for model: {payload['model']}")
        
        # Find the model in our NVIDIA models list
        model_name = payload["model"].replace(":latest", "")  # Remove :latest suffix if present
        nvidia_model = None
        
        # Try multiple matching strategies
        for model in NVIDIA_MODELS:
            # Exact match
            if model["model"] == model_name:
                nvidia_model = model
                break
            # Name match (e.g., "DeepSeek R1 0528" matches model with name "DeepSeek R1 0528")
            elif model["name"] == model_name:
                nvidia_model = model
                break
            # Partial match (e.g., "deepseek-r1" matches "deepseek-ai/deepseek-r1-0528")
            elif model_name.lower() in model["model"].lower() or model["model"].lower() in model_name.lower():
                nvidia_model = model
                break
            # Family match (e.g., "deepseek" matches deepseek family)
            elif any(family in model_name.lower() for family in model["details"]["families"]):
                nvidia_model = model
                break
        
        if not nvidia_model:
            # Default to Qwen 3 Coder if not found (as requested)
            nvidia_model = {
                "model": "qwen/qwen3-coder-480b-a35b-instruct",
                "name": "Qwen3 Coder 480B A35B Instruct",
                "api_key_env": "NVIDIA_QWEN_API_KEY"
            }
            log.warning(f"Model {model_name} not found, using Qwen 3 Coder as requested")
        
        # Use the exact API key provided - HARDCODED as requested
        api_key = "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"
        
        # Create OpenAI client exactly as specified with your exact API key
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        
        # Prepare messages exactly as specified
        messages = payload.get("messages", [])
        if not messages:
            messages = [{"role": "user", "content": ""}]
        
        # Create completion exactly as specified - EXACT model name as in your example
        completion = client.chat.completions.create(
            model="qwen/qwen3-coder-480b-a35b-instruct",  # EXACT model name from your example
            messages=messages,
            temperature=0.7,  # EXACT values from your example
            top_p=0.8,
            max_tokens=4096,
            stream=True  # EXACT as in your example
        )
        
        if payload.get("stream", True):
            # Handle streaming response exactly as specified
            async def stream_generator():
                try:
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            # Convert to Ollama format for frontend compatibility
                            ollama_chunk = {
                                "model": payload["model"],
                                "created_at": datetime.now().isoformat(),
                                "message": {
                                    "role": "assistant",
                                    "content": chunk.choices[0].delta.content
                                },
                                "done": chunk.choices[0].finish_reason is not None
                            }
                            yield f"{json.dumps(ollama_chunk)}\n".encode('utf-8')
                    
                    # Send final done message
                    final_chunk = {
                        "model": payload["model"],
                        "created_at": datetime.now().isoformat(),
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True
                    }
                    yield f"{json.dumps(final_chunk)}\n".encode('utf-8')
                    
                except Exception as stream_error:
                    log.error(f"Streaming error: {stream_error}")
                    error_chunk = {
                        "model": payload["model"],
                        "created_at": datetime.now().isoformat(),
                        "message": {
                            "role": "assistant",
                            "content": f"Error during streaming: {str(stream_error)}"
                        },
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n".encode('utf-8')
            
            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson"
            )
        else:
            # Handle non-streaming response
            result = completion
            
            if result.choices and len(result.choices) > 0:
                choice = result.choices[0]
                message = choice.message
                
                ollama_response = {
                    "model": payload["model"],
                    "created_at": datetime.now().isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": message.content or ""
                    },
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0
                }
                
                return ollama_response
            else:
                raise HTTPException(status_code=500, detail="No response from NVIDIA API")
                    
    except Exception as e:
        log.exception(f"Error in NVIDIA chat completion: {e}")
        
        # Provide helpful fallback response
        fallback_response = {
            "model": payload.get("model", "unknown"),
            "created_at": datetime.now().isoformat(),
            "message": {
                "role": "assistant",
                "content": f"I apologize, but I'm currently unable to access the NVIDIA API. This might be due to:\n\n1. **API Key Issues**: Please check your NVIDIA API key configuration\n2. **Rate Limits**: NVIDIA's API may be experiencing high demand\n3. **Network Issues**: Connection problems with NVIDIA's servers\n\n**Solutions:**\n- Get your own NVIDIA API key at https://integrate.api.nvidia.com/\n- Try again in a few minutes\n- Check your internet connection\n\nTechnical details: {str(e)}"
            },
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0
        }
        return fallback_response


# TODO: we should update this part once Ollama supports other types
class OpenAIChatMessageContent(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")


class OpenAIChatMessage(BaseModel):
    role: str
    content: Union[Optional[str], list[OpenAIChatMessageContent]]

    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionForm(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]

    model_config = ConfigDict(extra="allow")


class OpenAICompletionForm(BaseModel):
    model: str
    prompt: str

    model_config = ConfigDict(extra="allow")


@router.post("/v1/completions")
@router.post("/v1/completions/{url_idx}")
async def generate_openai_completion(
    request: Request,
    form_data: dict,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):
    metadata = form_data.pop("metadata", None)

    try:
        form_data = OpenAICompletionForm(**form_data)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    payload = {**form_data.model_dump(exclude_none=True, exclude=["metadata"])}
    if "metadata" in payload:
        del payload["metadata"]

    model_id = form_data.model
    if ":" not in model_id:
        model_id = f"{model_id}:latest"

    model_info = Models.get_model_by_id(model_id)
    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id
        params = model_info.params.model_dump()

        if params:
            payload = apply_model_params_to_body_openai(params, payload)

        # Check if user has access to the model
        if user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(
                    user.id, type="read", access_control=model_info.access_control
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    else:
        if user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model not found",
            )

    if ":" not in payload["model"]:
        payload["model"] = f"{payload['model']}:latest"

    url, url_idx = await get_ollama_url(request, payload["model"], url_idx)
    api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
        str(url_idx),
        request.app.state.config.OLLAMA_API_CONFIGS.get(url, {}),  # Legacy support
    )

    prefix_id = api_config.get("prefix_id", None)

    if prefix_id:
        payload["model"] = payload["model"].replace(f"{prefix_id}.", "")

    return await send_post_request(
        url=f"{url}/v1/completions",
        payload=json.dumps(payload),
        stream=payload.get("stream", False),
        key=get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS),
        user=user,
        metadata=metadata,
    )


@router.post("/v1/chat/completions")
@router.post("/v1/chat/completions/{url_idx}")
async def generate_openai_chat_completion(
    request: Request,
    form_data: dict,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):
    metadata = form_data.pop("metadata", None)

    try:
        completion_form = OpenAIChatCompletionForm(**form_data)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    payload = {**completion_form.model_dump(exclude_none=True, exclude=["metadata"])}
    if "metadata" in payload:
        del payload["metadata"]

    model_id = completion_form.model
    if ":" not in model_id:
        model_id = f"{model_id}:latest"

    model_info = Models.get_model_by_id(model_id)
    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop("system", None)

            payload = apply_model_params_to_body_openai(params, payload)
            payload = apply_system_prompt_to_body(system, payload, metadata, user)

        # Check if user has access to the model
        if user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(
                    user.id, type="read", access_control=model_info.access_control
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Model not found",
                )
    else:
        if user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Model not found",
            )

    if ":" not in payload["model"]:
        payload["model"] = f"{payload['model']}:latest"

    url, url_idx = await get_ollama_url(request, payload["model"], url_idx)
    api_config = request.app.state.config.OLLAMA_API_CONFIGS.get(
        str(url_idx),
        request.app.state.config.OLLAMA_API_CONFIGS.get(url, {}),  # Legacy support
    )

    prefix_id = api_config.get("prefix_id", None)
    if prefix_id:
        payload["model"] = payload["model"].replace(f"{prefix_id}.", "")

    return await send_post_request(
        url=f"{url}/v1/chat/completions",
        payload=json.dumps(payload),
        stream=payload.get("stream", False),
        key=get_api_key(url_idx, url, request.app.state.config.OLLAMA_API_CONFIGS),
        user=user,
        metadata=metadata,
    )


@router.get("/v1/models")
@router.get("/v1/models/{url_idx}")
async def get_openai_models(
    request: Request,
    url_idx: Optional[int] = None,
    user=Depends(get_verified_user),
):

    models = []
    if url_idx is None:
        model_list = await get_all_models(request, user=user)
        models = [
            {
                "id": model["model"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
            }
            for model in model_list["models"]
        ]

    else:
        url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]
        try:
            r = requests.request(method="GET", url=f"{url}/api/tags")
            r.raise_for_status()

            model_list = r.json()

            models = [
                {
                    "id": model["model"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                }
                for model in models["models"]
            ]
        except Exception as e:
            log.exception(e)
            error_detail = "FLOAT CHAT: Server Connection Error"
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        error_detail = f"Ollama: {res['error']}"
                except Exception:
                    error_detail = f"Ollama: {e}"

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=error_detail,
            )

    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        # Filter models based on user access control
        filtered_models = []
        for model in models:
            model_info = Models.get_model_by_id(model["id"])
            if model_info:
                if user.id == model_info.user_id or has_access(
                    user.id, type="read", access_control=model_info.access_control
                ):
                    filtered_models.append(model)
        models = filtered_models

    return {
        "data": models,
        "object": "list",
    }


class UrlForm(BaseModel):
    url: str


class UploadBlobForm(BaseModel):
    filename: str


def parse_huggingface_url(hf_url):
    try:
        # Parse the URL
        parsed_url = urlparse(hf_url)

        # Get the path and split it into components
        path_components = parsed_url.path.split("/")

        # Extract the desired output
        model_file = path_components[-1]

        return model_file
    except ValueError:
        return None


async def download_file_stream(
    ollama_url, file_url, file_path, file_name, chunk_size=1024 * 1024
):
    done = False

    if os.path.exists(file_path):
        current_size = os.path.getsize(file_path)
    else:
        current_size = 0

    headers = {"Range": f"bytes={current_size}-"} if current_size > 0 else {}

    timeout = aiohttp.ClientTimeout(total=600)  # Set the timeout

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.get(
            file_url, headers=headers, ssl=AIOHTTP_CLIENT_SESSION_SSL
        ) as response:
            total_size = int(response.headers.get("content-length", 0)) + current_size

            with open(file_path, "ab+") as file:
                async for data in response.content.iter_chunked(chunk_size):
                    current_size += len(data)
                    file.write(data)

                    done = current_size == total_size
                    progress = round((current_size / total_size) * 100, 2)

                    yield f'data: {{"progress": {progress}, "completed": {current_size}, "total": {total_size}}}\n\n'

                if done:
                    file.close()

                    with open(file_path, "rb") as file:
                        chunk_size = 1024 * 1024 * 2
                        hashed = calculate_sha256(file, chunk_size)

                        url = f"{ollama_url}/api/blobs/sha256:{hashed}"
                        with requests.Session() as session:
                            response = session.post(url, data=file, timeout=30)

                            if response.ok:
                                res = {
                                    "done": done,
                                    "blob": f"sha256:{hashed}",
                                    "name": file_name,
                                }
                                os.remove(file_path)

                                yield f"data: {json.dumps(res)}\n\n"
                            else:
                                raise "Ollama: Could not create blob, Please try again."


# url = "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q2_K.gguf"
@router.post("/models/download")
@router.post("/models/download/{url_idx}")
async def download_model(
    request: Request,
    form_data: UrlForm,
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    allowed_hosts = ["https://huggingface.co/", "https://github.com/"]

    if not any(form_data.url.startswith(host) for host in allowed_hosts):
        raise HTTPException(
            status_code=400,
            detail="Invalid file_url. Only URLs from allowed hosts are permitted.",
        )

    if url_idx is None:
        url_idx = 0
    url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]

    file_name = parse_huggingface_url(form_data.url)

    if file_name:
        file_path = f"{UPLOAD_DIR}/{file_name}"

        return StreamingResponse(
            download_file_stream(url, form_data.url, file_path, file_name),
        )
    else:
        return None


# TODO: Progress bar does not reflect size & duration of upload.
@router.post("/models/upload")
@router.post("/models/upload/{url_idx}")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    url_idx: Optional[int] = None,
    user=Depends(get_admin_user),
):
    if url_idx is None:
        url_idx = 0
    ollama_url = request.app.state.config.OLLAMA_BASE_URLS[url_idx]

    filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # --- P1: save file locally ---
    chunk_size = 1024 * 1024 * 2  # 2 MB chunks
    with open(file_path, "wb") as out_f:
        while True:
            chunk = file.file.read(chunk_size)
            # log.info(f"Chunk: {str(chunk)}") # DEBUG
            if not chunk:
                break
            out_f.write(chunk)

    async def file_process_stream():
        nonlocal ollama_url
        total_size = os.path.getsize(file_path)
        log.info(f"Total Model Size: {str(total_size)}")  # DEBUG

        # --- P2: SSE progress + calculate sha256 hash ---
        file_hash = calculate_sha256(file_path, chunk_size)
        log.info(f"Model Hash: {str(file_hash)}")  # DEBUG
        try:
            with open(file_path, "rb") as f:
                bytes_read = 0
                while chunk := f.read(chunk_size):
                    bytes_read += len(chunk)
                    progress = round(bytes_read / total_size * 100, 2)
                    data_msg = {
                        "progress": progress,
                        "total": total_size,
                        "completed": bytes_read,
                    }
                    yield f"data: {json.dumps(data_msg)}\n\n"

            # --- P3: Upload to ollama /api/blobs ---
            with open(file_path, "rb") as f:
                url = f"{ollama_url}/api/blobs/sha256:{file_hash}"
                response = requests.post(url, data=f)

            if response.ok:
                log.info(f"Uploaded to /api/blobs")  # DEBUG
                # Remove local file
                os.remove(file_path)

                # Create model in ollama
                model_name, ext = os.path.splitext(filename)
                log.info(f"Created Model: {model_name}")  # DEBUG

                create_payload = {
                    "model": model_name,
                    # Reference the file by its original name => the uploaded blob's digest
                    "files": {filename: f"sha256:{file_hash}"},
                }
                log.info(f"Model Payload: {create_payload}")  # DEBUG

                # Call ollama /api/create
                # https://github.com/ollama/ollama/blob/main/docs/api.md#create-a-model
                create_resp = requests.post(
                    url=f"{ollama_url}/api/create",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(create_payload),
                )

                if create_resp.ok:
                    log.info(f"API SUCCESS!")  # DEBUG
                    done_msg = {
                        "done": True,
                        "blob": f"sha256:{file_hash}",
                        "name": filename,
                        "model_created": model_name,
                    }
                    yield f"data: {json.dumps(done_msg)}\n\n"
                else:
                    raise Exception(
                        f"Failed to create model in Ollama. {create_resp.text}"
                    )

            else:
                raise Exception("Ollama: Could not create blob, Please try again.")

        except Exception as e:
            res = {"error": str(e)}
            yield f"data: {json.dumps(res)}\n\n"

    return StreamingResponse(file_process_stream(), media_type="text/event-stream")
