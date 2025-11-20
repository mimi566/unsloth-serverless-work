import os
import runpod
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import time
from typing import AsyncGenerator

# ─────────────────────────────────────────────────────────────
# Load model once at worker startup (Unsloth 4-bit)
# ─────────────────────────────────────────────────────────────
print("Loading Unsloth model - this takes 10-15 seconds on first cold start...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=os.getenv("MODEL_NAME", "Sourabh66/Llama-2-17B-Fine-Tune-Blog"),
    max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "32768")),
    dtype=None,                    # Auto (bfloat16 on A100/H100)
    load_in_4bit=True,
    device_map="auto",
)

# Enable 2–4× faster inference
FastLanguageModel.for_inference(model)
print("Model loaded successfully with Unsloth! Ready for inference.")

# ─────────────────────────────────────────────────────────────
# Human-like slow streamer (your exact 0.028s per char)
# ─────────────────────────────────────────────────────────────
class HumanLikeStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if os.getenv("SLOW_STREAM", "true").lower() == "true":
            for char in text:
                print(char, end="", flush=True)
                time.sleep(0.028)  # Your exact human typing speed
        else:
            print(text, end="", flush=True)
        if stream_end:
            print(flush=True)

streamer = HumanLikeStreamer(tokenizer, skip_prompt=True)

# ─────────────────────────────────────────────────────────────
# Your full system prompt (baked in - perfect for blogging)
# ─────────────────────────────────────────────────────────────
HUMAN_VS_AI_ANALYZE = """
Analyze 1

Human-Written Version
Text:
    If you are a game lover and disappointed that you cannot play a single word in wordle game per day . Then , Wordle unlimited provides you a solution to your this problems

AI-Generated Version
Text:
    If you're an avid gamer who's frustrated by the limitation of playing just one word per day in the Wordle game, Wordle Unlimited offers a perfect solution to this issue.
"""

SOURCE_EXAMPLE = """
**GB WhatsApp APK**

<p>GBWhatsApp is a top choice for people who want extra features beyond regular WhatsApp. </p>
"""

SYSTEM_PROMPT = f"""
You are my personal *humanizer blogger model*.

STEP 1: First, analyze the HUMAN vs AI difference carefully:
{HUMAN_VS_AI_ANALYZE}

STEP 2: Now, model your entire blog tone and structure after this real human writing:
{SOURCE_EXAMPLE}

STEP 3: When generating, create a FULL blog post version of the given topic — not just rewriting the paragraph.

RULES:
- Output must feel like a complete, finished blog article ready for publishing.
- Use natural paragraphs, headings, bullet points when needed.
- Never sound robotic or generic.
- Always write in first-person or engaging blog style.
""".strip()

# ─────────────────────────────────────────────────────────────
# Main handler (OpenAI compatible + streaming)
# ─────────────────────────────────────────────────────────────
async def handler(job) -> AsyncGenerator[str, None]:
    try:
        job_input = job["input"]
        messages = job_input.get("messages", [])
        if not messages:
            yield '{"error": "No messages provided"}'
            return

        # Inject system prompt if not present
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": job_input.get("max_tokens", 2400),
            "temperature": job_input.get("temperature", 1.25),
            "top_p": job_input.get("top_p", 0.92),
            "repetition_penalty": 1.05,
            "do_sample": True,
            "streamer": streamer if job_input.get("stream", True) else None,
            "use_cache": True,
        }

        # Streaming mode (RunPod handles chunking automatically)
        if job_input.get("stream", True):
            def generate_stream():
                model.generate(**generation_kwargs)
                yield ""  # Final yield to close stream

            for _ in generate_stream():
                # RunPod auto-collects stdout chunks from streamer
                pass
            return

        # Non-streaming (rare)
        output = model.generate(**generation_kwargs)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        text = text[len(prompt):].strip()

        yield {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        }

    except Exception as e:
        yield f'{{"error": "Inference failed: {str(e)}"}}'

# ─────────────────────────────────────────────────────────────
# Start serverless worker
# ─────────────────────────────────────────────────────────────
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})