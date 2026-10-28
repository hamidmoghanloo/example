"""Serve Qwen3-4B text generation over HTTP."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-4B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

tokenizer = None
model = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model before serving requests."""
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_DTYPE,
        use_safetensors=True,
    )
    model.to(DEVICE)
    model.eval()

    yield

    model = None
    tokenizer = None


app = FastAPI(
    title="Aptar Qwen3 API",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=48, ge=1, le=128)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok" if model is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, str | int]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model is loading")

    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][prompt_length:]
    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return {
        "model": MODEL_ID,
        "device": DEVICE,
        "prompt": request.prompt,
        "generated_text": generated_text,
        "generated_tokens": int(generated_tokens.shape[0]),
    }
