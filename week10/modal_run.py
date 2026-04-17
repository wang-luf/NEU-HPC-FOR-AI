"""Run nano-sglang on Modal.

Usage:
    modal run modal_run.py::run      # run the example
    modal run modal_run.py::test     # run tests
"""

import modal

MODEL_NAME = "Qwen/Qwen3-0.6B"

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "huggingface_hub", "pytest", "accelerate")
    .run_function(download_model)
    .add_local_dir("nano_sglang", remote_path="/root/nano_sglang")
    .add_local_dir("tests", remote_path="/root/tests")
)

app = modal.App("nano-sglang")

@app.function(image=image, gpu="A100-40GB", timeout=600)
def run():
    """Run the example."""
    from nano_sglang.engine import Engine
    from nano_sglang.sampling import SamplingParams

    engine = Engine(MODEL_NAME)
    params = SamplingParams(temperature=0, max_tokens=50)
    output = engine.generate("The capital of France is", params)
    print(f"Output: {output}")

@app.function(image=image, gpu="A100-40GB", timeout=600)
def test():
    import subprocess
    subprocess.run(["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"], check=False)
