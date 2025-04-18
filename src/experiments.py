from itertools import islice
import torch
import multiprocessing as mp
import time

# testing encode decode is the same for my decoder
    # arr = np.fromfile(OUTPUT_FILE, dtype=np.uint16)
    # decoded = tokenizer.decode(arr)
    # with open(DATA_PATH / f"{ENCODED_FILE_NAME}.txt", "r") as f:
    #     content = f.read()
    # assert decoded == content

# process line by line, but we have chunks for file encoding
    # tokens = []
    # for line in chunk.splitlines(keepends=True):
    #     tokens.extend(tokenizer.encode(line))
    # return tokens

def process_line(lines):
    # Do something with the line
    return len(lines)  # For example, count the line

def chunked_file(f, chunk_size=10000):
    while True:
        chunk = list(islice(f, chunk_size))
        if not chunk:
            break
        yield chunk

def main():

    # l = [1, 2, 3]
    # for e in iter(lambda: l[:1], []):
    #     print(e)

    file_path = "/Users/bsun/Documents/Grad School/LLMs/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    num_workers = mp.cpu_count() * 2  # Good starting point for I/O-bound work

    start_time = time.perf_counter()

    with mp.Pool(processes=num_workers) as pool:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            results = pool.imap_unordered(process_line, chunked_file(f, chunk_size=30000), chunksize=5)
            total = sum(results)

    end_time = time.perf_counter()
    print(f"Time: {end_time-start_time}")

    print(f"Total processed lines: {total}")

if __name__ == '__main__':
    import torch.utils.benchmark as benchmark
    tensor = torch.rand(1000000, device="mps")

    timer = benchmark.Timer(
        stmt="torch.sqrt(tensor)",
        setup="import torch; tensor = torch.rand(1000000, device='mps')",
        globals={"tensor": tensor},
    )
    print(timer.timeit(100))  # Benchmark over 100 iterations