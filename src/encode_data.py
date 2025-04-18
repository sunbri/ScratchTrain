import cProfile
from functools import partial
import io
from multiprocessing import Pool, cpu_count
import numpy as np
import pathlib
import pickle
import pstats
import subprocess
from typing import BinaryIO

import time

from structures.tokenizer import Tokenizer, train_bpe, find_chunk_boundaries

PARENT_PATH = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PARENT_PATH / "data"
OUTPUT_PATH = PARENT_PATH / "output"
BPE_FILE_NAME = "TinyStoriesV2-GPT4-train"
ENCODED_FILE_NAME = "TinyStoriesV2-GPT4-train"

OUTPUT_FILE = OUTPUT_PATH / f"encoded_{ENCODED_FILE_NAME}.bin"

def make_bpe(profile=False):
    pr = cProfile.Profile()
    if profile:
        pr.enable()
    vocab, merges = train_bpe(DATA_PATH / f"{BPE_FILE_NAME}.txt", 10000, ['<|endoftext|>'])

    with open(f"{OUTPUT_PATH}/vocab/vocab_{BPE_FILE_NAME}.txt", 'w') as f:
        f.write(str(vocab))
    with open(f"{OUTPUT_PATH}/vocab/vocab_{BPE_FILE_NAME}.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    with open(f"{OUTPUT_PATH}/merges/merges_{BPE_FILE_NAME}.txt", 'w') as f:
        for merge in merges:
            f.write(f"{merge}\n")
    with open(f"{OUTPUT_PATH}/merges/merges_{BPE_FILE_NAME}.pkl", 'wb') as f:
        pickle.dump(merges, f)

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        with open(f"{OUTPUT_PATH}/profiles/profile_{BPE_FILE_NAME}.txt", 'w') as f:
            f.write(s.getvalue())

def chunked_file_tokenizer(f: BinaryIO, boundaries, tokenizer):
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
        yield (tokenizer, chunk)

def chunked_tokens(input_tuple):
    tokenizer, chunk = input_tuple
    return tokenizer.encode(chunk)

def load_and_tokenize(special_tokens: list[str]):
    with open(f"{OUTPUT_PATH}/vocab/vocab_{BPE_FILE_NAME}.pkl", 'rb') as f:
        vocab = pickle.load(f)
    with open(f"{OUTPUT_PATH}/merges/merges_{BPE_FILE_NAME}.pkl", 'rb') as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)

    max_length = int(subprocess.check_output(f"/usr/bin/wc -c data/{ENCODED_FILE_NAME}.txt",
                                             shell=True).split()[0])
    data_type = np.dtype('uint16')
    # this is not TECHNICALLY enough since UTF-8 can map one character to up to 4 bytes
    # but assuming we're gonna be using English it should be fine
    output_bin = np.memmap(OUTPUT_FILE, dtype=data_type, mode='w+', shape=(max_length,))
    encoded_special_tokens = [t.encode('utf-8') for t in special_tokens]
    assert len(encoded_special_tokens) == 1
    
    num_processes = cpu_count() * 2
    with Pool(num_processes) as pool:
        with open(DATA_PATH / f"{ENCODED_FILE_NAME}.txt", "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, encoded_special_tokens[0])
            results = pool.imap(chunked_tokens, chunked_file_tokenizer(f, boundaries, tokenizer), 
                                chunksize=1)
            idx = 0
            for result in results:
                size = len(result)
                output_bin[idx:idx+size] = result
                idx += size

    # make sure to commit all changes
    output_bin.flush()
    del output_bin

    # rewrite the file 
    bytes_written = idx * data_type.itemsize
    with open(OUTPUT_FILE, 'r+b') as f:
        f.truncate(bytes_written)

if __name__ == "__main__":

    old_time = time.time()

    make_bpe(profile=False)
    # load_and_tokenize(special_tokens=['<|endoftext|>'])

    new_time = time.time()
    print(new_time - old_time)