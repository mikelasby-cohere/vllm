import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"

# with open(os.path.join(os.path.dirname(__file__), 'qwen_1m', "1m.txt")) as f:
with open(os.path.join(os.path.dirname(__file__), "qwen_1m", "64k.txt")) as f:
    prompt = f.read()

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
llm = LLM(model=os.path.expanduser("Qwen/Qwen2.5-7B-Instruct-1M"),
          gpu_memory_utilization=0.8,
          max_model_len=512000,
          tensor_parallel_size=1,
          enforce_eager=True,
          disable_custom_all_reduce=True,
          enable_chunked_prefill=True,
        #   max_num_batched_tokens=131072)
          max_num_batched_tokens=4096)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt_token_ids = output.prompt_token_ids
    generated_text = output.outputs[0].text
    print(f"Prompt length: {len(prompt_token_ids)}, "
          f"Generated text: {generated_text!r}")
