"""Serve a tiny Hugging Face text-generation model over HTTP."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "sshleifer/tiny-gpt2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Download the small model into the persistent workspace and load it."""
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    yield
    model = None
    tokenizer = None


app = FastAPI(title="Aptar Tiny Model API", version="1.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    """Validate the prompt and keep inference bounded for this demo."""

    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=48, ge=1, le=128)


@app.get("/health")
def health() -> dict[str, str]:
    """Expose a lightweight process-health endpoint."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, str | int]:
    """Generate a short continuation and return it as JSON."""
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
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return {
        "model": MODEL_ID,
        "device": DEVICE,
        "prompt": request.prompt,
        "generated_text": generated,
        "generated_tokens": int(output.shape[1] - inputs["input_ids"].shape[1]),
    }
