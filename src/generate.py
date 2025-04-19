from structures.models import TransformerLM, softmax
from structures.tokenizer import Tokenizer

import pathlib
import pickle
import time
import torch

PARENT_PATH = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_PATH = PARENT_PATH / "output"
BPE_FILE_NAME = "TinyStoriesV2-GPT4-train"
ENCODED_FILE_NAME = "TinyStoriesV2-GPT4-train"

END_OF_TEXT = '<|endoftext|>'

def get_next_token(logits: torch.Tensor, temperature=1.0, top_p: float=0.9):
    probs = softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # sum up and mask (key: use torch operations)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cum_probs > top_p
    # shift mask by one to keep the token that pushed us over
    mask[..., 1:] = mask[..., :-1].clone()
    # always have at least one token to sample
    mask[..., 0] = False
    sorted_probs[mask] = 0.0

    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sample_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sample_idx).item()

def generate_text(model: TransformerLM, prefix: list[int], end_id: int,
                  max_token_generate: int=100):

    prefix_tensor = torch.tensor(prefix)
    
    for _ in range(max_token_generate):
        prefix_cut = prefix_tensor[..., -model.context_length:]
        with torch.no_grad():
            # get logits of last token only
            logits = model(prefix_cut)[-1,:]

        next_token = get_next_token(logits)
        prefix.append(next_token)
        prefix_tensor = torch.tensor(prefix)

        if next_token == end_id:
            break
    return prefix

def get_prefixes_and_generate(model: TransformerLM, device: str):
    # put into evaluation mode
    model.eval()
    # just for inference
    torch.set_default_device(device)

    prefixes = ['There once was a greedy, greedy boy who played with a ball, and his name was Bitch']
    with open(f"{OUTPUT_PATH}/vocab/vocab_{BPE_FILE_NAME}.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open(f"{OUTPUT_PATH}/merges/merges_{BPE_FILE_NAME}.pkl", 'rb') as f:
        merges = pickle.load(f)

    t = Tokenizer(vocab, merges, special_tokens=[END_OF_TEXT])
    end_id = t.special_tokens[END_OF_TEXT]

    start_inference = time.time()

    res = []
    for prefix in prefixes:
        print(f"Generating with: {prefix}")
        # think of it as a batch of size 1
        encoded_prefix = t.encode(prefix)
        assert(len(encoded_prefix) <= model.context_length)
        res.append(generate_text(model, encoded_prefix, end_id))
    
    for encoded_string in res:
        print(t.decode(encoded_string))
        print(f"Number of tokens: {len(encoded_string)}")
    
    print(f"Seconds: {time.time() - start_inference}")