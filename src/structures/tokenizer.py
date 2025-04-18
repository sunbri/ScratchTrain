from collections import Counter
from functools import cmp_to_key, partial
from multiprocessing import Pool, cpu_count
from typing import BinaryIO
import numpy as np
import os
import pickle
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # dict[int, bytes]
        self.vocab = vocab
        # dict[tuple[bytes, bytes], int]
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        # order special tokens
        # dict[str, int]
        self.special_tokens = {}
        token_int = len(self.vocab)
        if special_tokens:
            # reverse for '<|endoftext|><|endoftext|>' coming before '<|endoftext|>'
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                encoded = token.encode('utf-8')
                if encoded in self.reversed_vocab:
                    self.special_tokens[token] = self.reversed_vocab[encoded]
                    # keep the text one too just in case
                    self.reversed_vocab[token] = self.reversed_vocab[encoded]
                else:
                    self.vocab[token_int] = encoded
                    self.reversed_vocab[token] = token_int
                    self.special_tokens[token] = token_int
                    token_int += 1

            # the capturing parentheses leave the splitter in
            self.special_splitter = re.compile(f"({"|".join(re.escape(s) for s in special_tokens)})")

    # for now, expect pickled things only
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        res = []
        special_split = [text]
        if self.special_tokens:
            special_split = re.split(self.special_splitter, text)

        for split in special_split:
            if split in self.special_tokens:
                res.append(self.reversed_vocab[split])
            else:
                pre_tokens = re.findall(PAT, split)
                for pre_token in pre_tokens:
                    b_pre_token = np.frombuffer(pre_token.encode('utf-8'), dtype='S1').tolist()

                    while len(b_pre_token) > 1:
                        pairs = zip(b_pre_token, b_pre_token[1:])
                        merge_candidate = min(pairs, key=lambda x: self.merges.get(x, float('inf')))
                        if merge_candidate in self.merges:
                            merge_bytes = merge_candidate[0] + merge_candidate[1]
                            new_pre_token = []
                            idx = 0
                            while idx < len(b_pre_token):
                                if idx < len(b_pre_token) - 1 and \
                                    (b_pre_token[idx], b_pre_token[idx+1]) == merge_candidate:
                                    new_pre_token.append(merge_bytes)
                                    idx += 2
                                else:
                                    new_pre_token.append(b_pre_token[idx])
                                    idx += 1
                            b_pre_token = new_pre_token
                        else:
                            break
                    res.extend([self.reversed_vocab[x] for x in b_pre_token])

        return res

    def encode_iterable(self, iterable):
        for e in iterable:
            yield from self.encode(e)

    def decode(self, ids: list[int]) -> str:
        res = b''.join([self.vocab[id] for id in ids])
        return res.decode('utf-8', errors='replace')

class pair_of_bytes:
    def __init__(self, pair: tuple[bytes], is_head: bool, is_dummy: bool, count: int):
        self.first = pair[0]
        self.second = pair[1]
        # counts the multiplicity of the pretoken this was spawned from
        self.count = count
        
        # for the linked list representing the pretoken itself
        # this will allow us to modify, within the pretoken, the
        # new pairs of bytes that arise from the merge
        self.is_head = is_head
        self.prev_token_pair: pair_of_bytes = None
        self.next_token_pair: pair_of_bytes = None

        # for the linked list representing in the locations
        # this will allow us to easily modify the list of
        # locations; will be a circular linked list with a dummy
        # because we will add things in one direction
        # and traverse in the reverse direction (to do the earliest first)
        self.is_dummy = is_dummy
        self.prev_location: pair_of_bytes = None
        self.next_location: pair_of_bytes = None

def bp_comparator(bp_item_1, bp_item_2):
    count_1, count_2 = bp_item_1[1], bp_item_2[1]
    if count_1 != count_2:
        return count_1 - count_2

    f_1, f_2 = bp_item_1[0][0], bp_item_2[0][0]
    if f_1 != f_2:
        return 1 if f_1 > f_2 else -1
    return 1 if bp_item_1[0][1] > bp_item_2[0][1] else -1

def chunked_file(f: BinaryIO, boundaries):
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
        yield chunk

def pre_tokenize(chunk, special_token):
    word_count = Counter()
    for split in chunk.split(special_token):
        word_count.update(PAT.findall(split))
    return word_count

def create_circular_list(node, count):
    dummy = pair_of_bytes((0, 0), False, True, count)
    dummy.next_location = node
    dummy.prev_location = node
    node.prev_location = dummy
    node.next_location = dummy
    return dummy

def insert_circular_list(node, dummy):
    node.prev_location = dummy
    node.next_location = dummy.next_location
    dummy.next_location.prev_location = node
    dummy.next_location = node

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):

    # initialize vocab
    vocab = {}
    id_count = 0
    # put the special tokens at the beginning of the vocabulary
    for token in special_tokens:
        vocab[id_count] = token.encode('utf-8')
        id_count += 1
    for i in range(256):
        vocab[id_count+i] = bytes([i])
    id_count = len(vocab)
    merges = []

    # locations: a dictionary that takes a literal pair of bytes and returns the 
    # locations in the pre_tokens of where they are
    locations = {}
    # bps: literal pairs of bytes -> count
    bps = {}

    total_count = Counter()
    # asserting that we only have "<|endoftext|>"
    assert len(special_tokens) == 1
    special_token = special_tokens[0]
    num_processes = cpu_count() * 4
    with Pool(num_processes) as pool:
        with open(input_path, "rb") as f:
            partial_pre_tokenize = partial(pre_tokenize, special_token=special_token)
            boundaries = find_chunk_boundaries(f, num_processes, special_token.encode('utf-8'))
            results = pool.imap_unordered(partial_pre_tokenize, chunked_file(f, boundaries))
            for result in results:
                total_count.update(result)
    
    for pre_token in total_count:
        b_pre_token = np.frombuffer(pre_token.encode('utf-8'), dtype='S1').tolist()
        # start the linked list of guys for pre_token, is_head
        curr_node = pair_of_bytes((0, 0), True, False, total_count[pre_token])
        for x, y in zip(b_pre_token, b_pre_token[1:]):
            # make linkage for each token
            node = pair_of_bytes((x, y), False, False, total_count[pre_token])
            curr_node.next_token_pair = node
            node.prev_token_pair = curr_node
            curr_node = node

            if (x,y) in bps:
                bps[(x,y)] += total_count[pre_token]
                dummy = locations[(x,y)]
                insert_circular_list(node, dummy)
            else:
                # create dummy for circular linked list
                locations[(x,y)] = create_circular_list(node, total_count[pre_token])
                bps[(x,y)] = total_count[pre_token]

    # take only what we need
    sorted_items = sorted(bps.items(), key=cmp_to_key(bp_comparator), reverse=True)
    bps = dict(sorted_items)

    # merging step
    while bps and len(vocab) < vocab_size:
        # just get the max; sorting and pruning bps can make sense
        # if the vocab grows really really big, but doesn't happen in our cases
        bp = max(bps.items(), key=cmp_to_key(bp_comparator))[0]
        if bp == 0:
            break
        merges.append(bp)
        vocab[id_count] = bp[0] + bp[1]
        id_count += 1

        # loop through and update counts for new thing
        location: pair_of_bytes = locations[bp].prev_location
        while not location.is_dummy:
            p = location.prev_token_pair
            n = location.next_token_pair
            if p and not p.is_head:
                if (p.first, p.second) in bps:
                    bps[(p.first, p.second)] -= location.count
                (p.prev_location.next_location) = p.next_location
                (p.next_location.prev_location) = p.prev_location
                p.second = bp[0] + bp[1]
                if (p.first, p.second) in locations:
                    bps[(p.first, p.second)] += location.count
                    dummy = locations[(p.first, p.second)]
                    insert_circular_list(p, dummy)
                else:
                    locations[(p.first, p.second)] = create_circular_list(p, location.count)
                    bps[(p.first, p.second)] = location.count
            if n:
                if (n.first, n.second) in bps:
                    bps[(n.first, n.second)] -= location.count
                (n.prev_location.next_location) = n.next_location
                (n.next_location.prev_location) = n.prev_location
                n.first = bp[0] + bp[1]
                if (n.first, n.second) in locations:
                    bps[(n.first, n.second)] += location.count
                    dummy = locations[(n.first, n.second)]
                    insert_circular_list(n, dummy)
                else:
                    locations[(n.first, n.second)] = create_circular_list(n, location.count)
                    bps[(n.first, n.second)] = location.count
                p.next_token_pair = n
                n.prev_token_pair = p
            else:
                p.next_token_pair = None
            
            # traverse backwards so we visit earlier locations first (important for '....')
            location = location.prev_location

        del bps[bp]

    return vocab, merges