# src/handler.py — FINAL WORKING VERSION (Unsloth + RunPod Streaming)
import os
import runpod
import torch
import logging
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
import time
from typing import Generator, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Handler script starting...")

# Global model (lazy loaded)
model = None
tokenizer = None

# Your full system prompt (keep exactly as you had)
SYSTEM_PROMPT = """
You are my personal *humanizer blogger model*.

STEP 1: First, analyze the HUMAN vs AI difference carefully:
Analyze 1

Human-Written Version
Text:
    If you are a game lover and disappointed that you cannot play a single word in wordle game per day . Then , Wordle unlimited provides you a solution to your this problems

AI-Generated Version
Text:
    If you're an avid gamer who's frustrated by the limitation of playing just one word per day in the Wordle game, Wordle Unlimited offers a perfect solution to this issue.

STEP 2: Now, model your entire blog tone and structure after this real human writing:
**GB WhatsApp APK**

<p>GBWhatsApp is a top choice for people who want extra features beyond regular WhatsApp. </p>

STEP 3: When generating, create a FULL blog post version of the given topic — not just rewriting the paragraph.

RULES:
- Output must feel like a complete, finished blog article ready for publishing.
- Use natural paragraphs, headings, bullet points when needed.
- Never sound robotic or generic.
- Always write in first-person or engaging blog style.
""".strip()

def load_model():
    global model, tokenizer
    if model is None:
        logger.info("Loading Unsloth model...")
        model_name = os.getenv("MODEL_NAME", "Sourabh66/Llama-2-17B-Fine-Tune-Blog")
        token = os.getenv("HF_TOKEN")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=32768,
            dtype=None,
            load_in_4bit=True,
            token=token,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully!")

# Parse input like vLLM's JobInput
def get_input_params(job_input: Dict[str, Any]):
    messages = job_input.get("messages", [])
    
    # Support raw "prompt" input for convenience
    if not messages:
        prompt = job_input.get("prompt")
        if prompt:
            messages = [{"role": "user", "content": prompt}]

    if not messages:
        raise ValueError("No messages provided. Please send 'messages' (list) or 'prompt' (string) in your input.")

    max_tokens = job_input.get("max_tokens", 2400)
    temperature = job_input.get("temperature", 1.25)
    top_p = job_input.get("top_p", 0.92)
    stream = job_input.get("stream", True)

    return messages, max_tokens, temperature, top_p, stream

# Sync handler — yields OpenAI-compatible chunks
def handler(job) -> Generator[Dict, None, None]:
    try:
        load_model()
        job_input = job["input"]
        messages, max_tokens, temperature, top_p, stream = get_input_params(job_input)

        # Inject system prompt if not present
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Format prompt using chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": 1.05,
            "use_cache": True,
        }

        if stream:
            # Use TextIteratorStreamer for real-time streaming
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer

            # Run generation in a separate thread so we can yield chunks
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens as they are generated
            for new_text in streamer:
                yield {
                    "choices": [{
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }]
                }
            
            # Wait for thread to finish (should be done by now)
            thread.join()

            # Final chunk
            yield {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }

        else:
            # Non-streaming: return full text
            with torch.no_grad():
                output_ids = model.generate(**generation_kwargs)
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response_text = full_text[len(prompt):].strip()

            yield {
                "choices": [{
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }]
            }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        yield {"error": str(e)}

# Start RunPod serverless
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": False,   # Must be False for real-time streaming
})