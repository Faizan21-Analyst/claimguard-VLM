"""Hugging Face ZeroGPU Space: the ClaimGuard fine-tuned vision model as a small web API.

One endpoint, /analyze: a photo in, the model's raw text out. The ClaimGuard API calls it with gradio_client
(src/claimguard/vlm/remote_backend.py) and does the JSON parsing and validation itself.

All model code lives in inference.py, which knows nothing about Gradio, so moving to another host later means
replacing only this file.
"""
import gradio as gr
import spaces

from inference import generate, load_model

DEVICE = "cuda"

# ZeroGPU allows moving the model to "cuda" at start-up; a GPU is only attached while analyze() runs.
model, processor = load_model(DEVICE)


@spaces.GPU(duration=30)   # seconds of GPU one call may use; short = less quota spent and shorter queue
def analyze(image_path: str) -> str:
    return generate(model, processor, image_path, DEVICE)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath", label="Photo"),   # "filepath": the file is read as sent, no re-encoding
    outputs=gr.Textbox(label="Raw model output"),
    api_name="analyze",
    title="ClaimGuard vision model",
    description="Fine-tuned Qwen3-VL-2B. Returns a JSON description of visible home damage. Not a coverage decision.",
)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
