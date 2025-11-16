# app.py
import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import textwrap

# -------------------------
# Config - change MODEL_DIR if needed
# -------------------------
MODEL_DIR = "model"  # <-- change to your model folder
MAX_NEW_TOKENS = 180

# Reduce noisy tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Helper: load model & tokenizer once (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir: str):
    """
    Loads tokenizer and model. Tries AutoModelForCausalLM first.
    Returns (tokenizer, model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Ensure pad token exists and set left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Try to load model. Support plain HF model or PEFT-wrapped.
    model = None
    try:
        # Try to load full model (works if directory contains full model weights)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None)
    except Exception as e_full:
        # Fallback: try loading base model then apply PEFT adapter
        # This branch expects a base model file to exist or will raise clearly.
        try:
            base = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True)
            model = base
        except Exception:
            # Final fallback: raise the original error for clarity
            raise RuntimeError(f"Failed to load model from {model_dir}. Error: {e_full}")

    # Move model to device if not already sharded/device-mapped
    if not getattr(model, "is_loaded", False):
        model.to(device)

    # Make sure model config knows pad/eos ids
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model, device


# -------------------------
# Generation helper
# -------------------------
def generate_cpp_from_pseudo(tokenizer, model, device, pseudo: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Generate code given a short pseudo-code prompt string.
    Uses beam search + ngram blocking for cleaner outputs.
    """
    prompt = textwrap.dedent(f"""\
    ### PSEUDO-CODE:
    {pseudo.strip()}

    ### C++:
    """)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # guard
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)

    # Postprocess: extract only the C++ portion
    if "### C++:" in decoded:
        code = decoded.split("### C++:")[-1]
        # stop at next section marker, if present
        if "###" in code:
            code = code.split("###")[0]
    else:
        # fallback: use the tail of decoded text
        code = decoded

    return code.strip()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Pseudo→C++ (GPT-2) — Demo", layout="centered")

st.title("Pseudo → C++ — quick demo")
st.caption("Simple interface to test a fine-tuned GPT-2. Model loads once at startup.")

# Load model with spinner (hidden step)
load_placeholder = st.empty()
with load_placeholder:
    with st.spinner("Loading model and tokenizer (this runs once)…"):
        try:
            tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)
            load_placeholder.success("Model loaded — ready!")
        except Exception as e:
            load_placeholder.error(f"Failed to load model: {e}")
            st.stop()

# Examples area (very simple)
st.markdown("### Examples")
cols = st.columns(3)
examples = [
    'Print "Hello".',
    "Input an integer n. Print n.",
    "Input two integers a and b. Print a + b.",
    'Input an integer n. If n is even print "even". Else print "odd".',
    "Input an integer n. For i from 1 to n print i.",
    "Read array of size n. Print the maximum element.",
]
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, key=f"ex{i}"):
        st.session_state["input_text"] = ex

# Input area
st.markdown("### Enter pseudo-code")
input_text = st.text_area("Pseudo-code", value=st.session_state.get("input_text", examples[0]), height=160)

# Generation controls (compact)
with st.expander("Generation settings (advanced)", expanded=False):
    beams = st.slider("Beams (num_beams)", 1, 8, 4, help="Higher -> more thorough search (slower)")
    ngram = st.slider("No repeat n-gram size", 1, 6, 3, help="Prevents repeated phrases")
    max_tokens = st.slider("Max new tokens", 16, 512, MAX_NEW_TOKENS)
    do_sample = st.checkbox("Do sampling (stochastic)", value=False)
    if do_sample:
        top_p = st.slider("top_p", 0.1, 1.0, 0.92)
        temp = st.slider("temperature", 0.1, 2.0, 0.8)
    else:
        top_p = None
        temp = None

# Generate button
if st.button("Generate C++"):
    if not input_text.strip():
        st.warning("Please enter a pseudo-code prompt.")
    else:
        with st.spinner("Generating..."):
            # call generation with user-selected settings
            gen_kwargs = dict(
                num_beams=beams,
                no_repeat_ngram_size=ngram,
                max_new_tokens=max_tokens,
                early_stopping=True,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs.update(dict(top_p=top_p, temperature=temp, top_k=50))

            # run generation (safe torch.no_grad)
            with torch.no_grad():
                prompt = textwrap.dedent(f"""\
                ### PSEUDO-CODE:
                {input_text.strip()}

                ### C++:
                """)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                out = model.generate(**inputs, **gen_kwargs)

            decoded = tokenizer.decode(out[0], skip_special_tokens=False)
            # extract C++ block
            if "### C++:" in decoded:
                code_block = decoded.split("### C++:")[-1]
                if "###" in code_block:
                    code_block = code_block.split("###")[0]
            else:
                code_block = decoded

            st.markdown("### Generated C++")
            st.code(code_block.strip(), language="cpp")
            # small footer
            st.caption("Tip: if the output is repetitive, try enabling sampling or increasing 'no repeat n-gram'.")

# Footer: simple instructions
st.markdown("---")
st.markdown("**Notes:** Model loaded from `" + MODEL_DIR + "`. If you want to change the model path, edit `MODEL_DIR` in the script.")
