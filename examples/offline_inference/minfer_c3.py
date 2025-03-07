import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "MINFERENCE_FLASH_ATTN"

with open(os.path.join(os.path.dirname(__file__), "qwen_1m", "64k.txt")) as f:
    prompt = f.read()

MODEL_PATH = "/root/cohere_ckpt/c3-7b/c3-7b-hf/hugging_face/sparse"
# Sample prompts.
prompts = [
    prompt,
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
    detokenize=True,
    max_tokens=256,
)

# Create an LLM.
llm = LLM(
    model=os.path.expanduser(MODEL_PATH),
    gpu_memory_utilization=0.9,
    max_model_len=131072,
    tensor_parallel_size=1,
    enforce_eager=True,
    disable_custom_all_reduce=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt_token_ids = output.prompt_token_ids
    generated_text = output.outputs[0].text
    print(
        f"\n\nPrompt length: {len(prompt_token_ids)}, "
        f"Generated text: {generated_text!r}\n\n"
    )
    break
