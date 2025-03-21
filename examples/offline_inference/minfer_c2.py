import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "MINFERENCE_FLASH_ATTN"

with open(os.path.join(os.path.dirname(__file__), "qwen_1m", "20k_cohere.txt")) as f:
    prompt = f.read()

MODEL_PATH = "/root/cohere_ckpt/sparse-r-32b/"
# MODEL_PATH = "/root/cohere_ckpt/c3-7b-hf/hugging_face/poseidon"
# Sample prompts.
prompts = [
    prompt,
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.,
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
    max_model_len=48000,
    tensor_parallel_size=1,
    enforce_eager=True,
    disable_custom_all_reduce=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    # max_num_batched_tokens=2**15,
    # max_num_seqs=1,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate([prompt]*2, sampling_params)
# Print the outputs.
for output in outputs:
    # print(f"Prompt:\n{prompt}")
    prompt_token_ids = output.prompt_token_ids
    generated_text = output.outputs[0].text
    print(
        f"\n\nPrompt length: {len(prompt_token_ids)}, "
        f"Generated text: {generated_text!r}\n\n"
    )
