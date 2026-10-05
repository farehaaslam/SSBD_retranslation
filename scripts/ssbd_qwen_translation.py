import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SSBD bias parameter

BETA = 0.10

# Maximum number of tokens to generate after divergence

MAX_NEW_TOKENS = 128

# Use greedy decoding to reproduce Algorithm 1's argmax behavior

DO_SAMPLE = False

# ============================================================

# Load model

# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    torch_dtype="auto",

    device_map="auto",

)

model.eval()

print("Model loaded.")

# ============================================================

# Helper: build translation prompt

# ============================================================

def build_prompt(source_text):

    """

    Creates the source-side prompt.

    We deliberately do NOT put the previous translation here.

    The previous translation is the speculative draft and is

    added separately later.

    """

    messages = [

        {

            "role": "system",

            "content": (

                "You are a translation model. "

                "Translate English to french. "

                "Output only the french translation."

            ),

        },

        {

            "role": "user",

            "content": source_text,

        },

    ]

    # add_generation_prompt=True gives us the point from which

    # the target translation starts.
    prompt = tokenizer.apply_chat_template(

        messages,

        tokenize=True,

        add_generation_prompt=True,

        enable_thinking=False,

        return_tensors="pt",

    )

    # Some Transformers/tokenizer versions return a BatchEncoding here

    # instead of a raw tensor. model.generate(input_ids=...) requires

    # the actual input_ids tensor.

    if hasattr(prompt, "input_ids"):

        prompt_ids = prompt.input_ids

    elif isinstance(prompt, dict):

        prompt_ids = prompt["input_ids"]

    else:

        prompt_ids = prompt

    if not isinstance(prompt_ids, torch.Tensor):

        prompt_ids = torch.as_tensor(prompt_ids)

    return prompt_ids.to(model.device)

# ============================================================

# STEP 2

#

# Process current source X^T

# and create KV cache

# ============================================================

@torch.inference_mode()

def prepare_source_cache(source_text):

    """

    Equivalent to:

    Let prompt <- create_input(X^T, <BOS>)

    run forward pass with cached states for it

    Returns:

        prompt_ids

        past_key_values

        first_logits

    first_logits are the logits used to predict y_0.

    """

    prompt_ids = build_prompt(source_text)

    outputs = model(

        input_ids=prompt_ids,

        use_cache=True,

    )

    # KV cache representing the source/prompt context

    past_key_values = outputs.past_key_values

    # The logits at the final prompt position predict the first

    # target token y_0.

    first_logits = outputs.logits[:, -1, :]

    return prompt_ids, past_key_values, first_logits

# ============================================================

# STEP 3

#

# Run one forward pass over the draft

# ============================================================

@torch.inference_mode()

def verify_draft(

    draft_ids,

    past_key_values,

    first_logits,

    beta=0.1,

):

    """

    Verify the previous output token-by-token.

    draft_ids:

        token IDs of Y^(T-1)

    Returns:

        accepted_count

        accepted_ids

        divergence_position

        logits

    """

    draft_ids = draft_ids.to(model.device)

    m = draft_ids.shape[1]

    # --------------------------------------------------------

    # We need probability distributions:

    #

    # p(0) = P(y0 | X)

    #

    # p(1) = P(y1 | X, y0)

    #

    # p(2) = P(y2 | X, y0, y1)

    #

    # ...

    #

    # first_logits already gives p(0).

    # -------------------------------------------------------- 

    # Feed the entire draft in one forward pass.

    # The source/prompt KV cache is reused here.

    draft_outputs = None

    if m > 0:

        draft_outputs = model(

            input_ids=draft_ids,

            past_key_values=past_key_values,

            use_cache=True,

        )

# --------------------------------------------------------

    # Construct p(0), p(1), ..., p(m-1)

    # --------------------------------------------------------

    probabilities = []

    # p(0)

    probabilities.append(first_logits)

    if m > 1:

        # draft_outputs.logits[:, :-1, :]

        # contains predictions for y1 ... y(m-1)

        for i in range(1, m):

            probabilities.append(

                draft_outputs.logits[:, i - 1, :]

            )

    # --------------------------------------------------------

    # Verification

    # --------------------------------------------------------

    accepted = []

    divergence_position = m

    for i in range(m):

        # Original model distribution

        logits_i = probabilities[i]

        # Convert logits → probabilities

        p_i = torch.softmax(logits_i, dim=-1)

        # Draft token y_i

        y_i = draft_ids[0, i]

        # ----------------------------------------------------

        # δ(. = y_i)

        #

        # One-hot distribution:

        #

        # y_i -> 1

        # everything else -> 0

        # ----------------------------------------------------

        biased_p = p_i * (1.0 - beta)

        biased_p[0, y_i] += beta

        # ----------------------------------------------------

        # argmax(p'(i))

        # ----------------------------------------------------

        predicted_token = torch.argmax(

            biased_p,

            dim=-1,

        ).item()

        # ----------------------------------------------------

        # Compare:

        #

        # y_i == argmax(p'(i)) ?

        # ----------------------------------------------------

        if predicted_token != y_i.item():

            divergence_position = i

            print(

                f"[DIVERGENCE] position={i}, "

                f"draft='{tokenizer.decode([y_i])}', "

                f"model='{tokenizer.decode([predicted_token])}'"

            )

            break

        # Token accepted

        accepted.append(y_i.item())

        print(

            f"[ACCEPT] position={i}, "

            f"token='{tokenizer.decode([y_i])}'"

        )

    accepted_ids = torch.tensor(

        accepted,

        dtype=torch.long,

        device=model.device,

    )

    return (

        len(accepted),

        accepted_ids,

        divergence_position,

    )

# ============================================================

# Generate suffix after divergence

# ============================================================

@torch.inference_mode()

def generate_suffix(

    source_text,

    accepted_ids,

    max_new_tokens=128,

):

    """

    Generate the new suffix after the divergence.

    For correctness and simplicity, this reconstructs the

    prompt + accepted prefix and starts normal autoregressive

    generation from there.

    This is slightly less optimized than manually preserving

    the target KV cache, but is much easier to verify first.

    """

    prompt_ids = build_prompt(source_text)

    if accepted_ids.numel() > 0:

        accepted_ids = accepted_ids.unsqueeze(0)

        input_ids = torch.cat(

            [

                prompt_ids,

                accepted_ids,

            ],

            dim=1,

        )

    else:

        input_ids = prompt_ids

    outputs = model.generate(

        input_ids=input_ids,

        max_new_tokens=max_new_tokens,

        do_sample=False,

        use_cache=True,

        pad_token_id=tokenizer.eos_token_id,

    )

    # Only return tokens generated after the accepted prefix

    new_tokens = outputs[:, input_ids.shape[1]:]

    return new_tokens[0]

# ============================================================

# COMPLETE SSBD STEP

# ============================================================

@torch.inference_mode()

def ssbd_step(

    current_source,

    previous_translation,

    beta=BETA,

):

    """

    One complete SSBD streaming step.

    Inputs:

        current_source:

            X^T

        previous_translation:

            Y^(T-1)

    Returns:

        current_translation:

            Y^T

    """

    print("\n" + "=" * 70)

    print("SSBD STEP")

    print("=" * 70)

    print("\nCURRENT SOURCE:")

    print(current_source)

    print("\nPREVIOUS TRANSLATION:")

    print(previous_translation)

    # --------------------------------------------------------

    # STEP 2

    # --------------------------------------------------------

    print("\n[STEP 2] Processing current source...")

    prompt_ids, source_cache, first_logits = (

        prepare_source_cache(current_source)

    )

    # --------------------------------------------------------

    # Convert previous translation into draft tokens

    # --------------------------------------------------------

    draft_ids = tokenizer(

        previous_translation,

        add_special_tokens=False,

        return_tensors="pt",

    ).input_ids.to(model.device)

    print("\nDRAFT TOKENS:")

    print(

        tokenizer.convert_ids_to_tokens(

            draft_ids[0].tolist()

        )

    )

    # --------------------------------------------------------

    # STEP 3-12

    # Verify draft

    # --------------------------------------------------------

    print("\n[STEP 3] Verifying draft...")

    accepted_count, accepted_ids, divergence = (

        verify_draft(

            draft_ids=draft_ids,

            past_key_values=source_cache,

            first_logits=first_logits,

            beta=beta,

        )

    )

    accepted_text = tokenizer.decode(

        accepted_ids,

        skip_special_tokens=True,

    )

    print("\nACCEPTED PREFIX:")

    print(repr(accepted_text))

    print(f"\nAccepted tokens: {accepted_count}")

    print(f"Divergence position: {divergence}")

    # --------------------------------------------------------

    # STEP 13+

    # Generate new suffix

    # --------------------------------------------------------

    print("\n[STEP 13+] Generating new suffix...")

    suffix_ids = generate_suffix(

        current_source,

        accepted_ids,

        max_new_tokens=MAX_NEW_TOKENS,

    )

    suffix_text = tokenizer.decode(

        suffix_ids,

        skip_special_tokens=True,

    )

    print("\nNEW SUFFIX:")

    print(repr(suffix_text))

    # --------------------------------------------------------

    # STEP 18

    #

    # Y^T = accepted prefix + newly decoded suffix

    # --------------------------------------------------------

    final_translation = (

        accepted_text + suffix_text

    )

    print("\nFINAL Y^T:")

    print(final_translation)

    return final_translation

# ============================================================

# Example streaming translation

# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------

    # First streaming chunk

    # --------------------------------------------------------

    source_1 = (

        "The company announced"

    )

    previous_translation = ""

    # Since there is no previous translation at T=0,

    # generate normally.

    prompt_ids = build_prompt(source_1)

    output = model.generate(

        input_ids=prompt_ids,

        max_new_tokens=40,

        do_sample=False,

        use_cache=True,

        pad_token_id=tokenizer.eos_token_id,

    )

    translation_1 = tokenizer.decode(

        output[0, prompt_ids.shape[1]:],

        skip_special_tokens=True,

    )

    print("\n")

    print("=" * 70)

    print("T = 1")

    print("=" * 70)

    print("SOURCE:")

    print(source_1)

    print("\nTRANSLATION:")

    print(translation_1)

    # --------------------------------------------------------

    # Second streaming chunk

    # --------------------------------------------------------

    source_2 = (

        "The company announced a new product today"

    )

    translation_2 = ssbd_step(

        current_source=source_2,

        previous_translation=translation_1,

        beta=0.10,

    )

    print("\n")

    print("=" * 70)

    print("RESULT")

    print("=" * 70)

    print("T1:", translation_1)

    print("T2:", translation_2)