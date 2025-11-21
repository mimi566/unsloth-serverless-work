import os
import runpod
import torch
import logging
from unsloth import FastLanguageModel
from transformers import TextStreamer
import time

# Setup RunPod logging (critical for debugging exit code 1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Handler script starting...")

# Global vars for model (lazy load on first request)
model = None
tokenizer = None
streamer = None
SYSTEM_PROMPT = """  # Your full prompt here — unchanged
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

HUMAN_VS_AI_ANALYZE = """..."""  # Your analyze text — unchanged
SOURCE_EXAMPLE = """..."""  # Your example — unchanged

def load_model():
    global model, tokenizer, streamer
    if model is None:
        logger.info("Loading Unsloth model...")
        token = os.getenv("HF_TOKEN")  # Optional HF token
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.getenv("MODEL_NAME", "Sourabh66/Llama-2-17B-Fine-Tune-Blog"),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "32768")),
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
            token=token,
        )
        FastLanguageModel.for_inference(model)
        logger.info("Model loaded successfully!")

        class HumanLikeStreamer(TextStreamer):
            def __init__(self, tokenizer, skip_prompt=True):
                super().__init__(tokenizer, skip_prompt=skip_prompt)

            def on_finalized_text(self, text: str, stream_end: bool = False):
                slow = os.getenv("SLOW_STREAM", "true").lower() == "true"
                if slow:
                    for char in text:
                        print(char, end="", flush=True)
                        time.sleep(0.028)
                else:
                    print(text, end="", flush=True)
                if stream_end:
                    print(flush=True)

        streamer = HumanLikeStreamer(tokenizer, skip_prompt=True)

# Sync handler (RunPod prefers sync for runsync; works with streaming)
def handler(job):
    try:
        load_model()  # Lazy load on first request
        job_input = job["input"]
        messages = job_input.get("messages", [])
        if not messages:
            return {"error": "No messages provided"}

        # Inject system prompt if missing
        if not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": job_input.get("max_tokens", 2400),
            "temperature": job_input.get("temperature", 1.25),
            "top_p": job_input.get("top_p", 0.92),
            "repetition_penalty": 1.05,
            "do_sample": True,
            "use_cache": True,
        }

        stream = job_input.get("stream", True)
        if stream:
            generation_kwargs["streamer"] = streamer
            output = model.generate(**generation_kwargs)
            # For streaming, RunPod aggregates stdout from streamer
            return {"status": "streaming_complete"}  # Dummy return for sync
        else:
            output = model.generate(**generation_kwargs)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            text = text[len(prompt):].strip()
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return {"error": f"Inference failed: {str(e)}"}

# Start serverless (sync handler + aggregate stream for compatibility)
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})