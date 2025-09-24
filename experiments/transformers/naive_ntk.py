import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple, Dict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel


def set_seed(seed: int) -> None:
    """we set seeds for reproducibility"""
    random.seed(seed)  # we fix python rng
    np.random.seed(seed)  # we fix numpy rng
    torch.manual_seed(seed)  # we fix torch cpu rng
    torch.cuda.manual_seed_all(seed)  # we fix torch gpu rng
    torch.backends.cudnn.deterministic = True  # we favor determinism
    torch.backends.cudnn.benchmark = False  # we disable autotune for determinism


def resolve_paths(this_file: Path) -> Tuple[Path, Path]:
    """we compute base output directories 'plots/transformers/orion_ntk' and 'data/transformers/orion_ntk'"""
    try:
        repo_root = this_file.resolve().parents[2]  # we go two levels up to reach repo root
    except Exception:
        repo_root = this_file.resolve().parent.parent  # we fallback to two parents up if structure is different
    plots_dir = repo_root / "plots" / "transformers" / "orion_ntk"  # we set plots path
    data_dir = repo_root / "data" / "transformers" / "orion_ntk"  # we set data path
    plots_dir.mkdir(parents=True, exist_ok=True)  # we ensure plots dir exists
    data_dir.mkdir(parents=True, exist_ok=True)  # we ensure data dir exists
    return plots_dir, data_dir  # we return directories


def load_hf_embeddings(model_name: str, device: torch.device) -> Tuple[AutoTokenizer, torch.Tensor]:
    """we load a huggingface tokenizer and the word embedding matrix"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # we load tokenizer
    base_model = AutoModel.from_pretrained(model_name)  # we load backbone
    base_model.eval()  # we put in eval
    with torch.no_grad():
        if hasattr(base_model, "embeddings") and hasattr(base_model.embeddings, "word_embeddings"):  # we check bert-like
            emb_matrix = base_model.embeddings.word_embeddings.weight.detach().clone()  # we get embeddings
        elif hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):  # we check some decoder-only
            emb_matrix = base_model.model.embed_tokens.weight.detach().clone()  # we get embeddings
        elif hasattr(base_model, "embed_tokens"):  # we check t5-like
            emb_matrix = base_model.embed_tokens.weight.detach().clone()  # we get embeddings
        else:
            raise RuntimeError("we could not locate a word embedding matrix in the chosen model")  # we raise error
    del base_model  # we free model
    torch.cuda.empty_cache()  # we free gpu cache
    emb_matrix = emb_matrix.to(device=device)  # we move embeddings to device
    return tokenizer, emb_matrix  # we return tokenizer and embeddings


def get_allowed_token_ids(tokenizer: AutoTokenizer, max_id: int) -> np.ndarray:
    """we build a mask of allowed token ids corresponding to whole-word BERT tokens (no '##', no specials, alphabetic)"""
    vocab = tokenizer.get_vocab()  # we get vocab mapping
    allowed: List[int] = []  # we init list
    specials = set(tokenizer.all_special_tokens)  # we special tokens
    for tok, tid in vocab.items():
        if tid >= max_id:  # we skip out-of-range ids
            continue  # we continue
        if tok in specials:  # we skip specials
            continue  # we continue
        if tok.startswith("##"):  # we skip subwords
            continue  # we continue
        if any(ch.isdigit() for ch in tok):  # we skip digits
            continue  # we continue
        if not any(ch.isalpha() for ch in tok):  # we require alphabetic
            continue  # we continue
        if len(tok) < 2:  # we skip too short
            continue  # we continue
        allowed.append(int(tid))  # we add id
    allowed = sorted(set(allowed))  # we unique+sort
    return np.array(allowed, dtype=np.int64)  # we return allowed ids


def filter_vocab_words(tokenizer: AutoTokenizer, max_words: int = 30000) -> Tuple[List[str], List[str], List[str]]:
    """we pick word-level tokens and categorize them by frequency for diverse sentences"""
    vocab_items = list(tokenizer.get_vocab().items())  # we list vocab
    vocab_items.sort(key=lambda kv: kv[1])  # we sort by id
    
    common_words: List[str] = []  # we frequent words
    medium_words: List[str] = []  # we medium words
    rare_words: List[str] = []  # we rare words
    
    for i, (tok, _) in enumerate(vocab_items[:max_words]):
        if tok.startswith("##"):  # we skip subword continuations
            continue  # we continue
        t = tok.strip()  # we clean
        if t in tokenizer.all_special_tokens:  # we skip special tokens
            continue  # we continue
        if any(ch.isdigit() for ch in t):  # we skip digits
            continue  # we continue
        if not any(ch.isalpha() for ch in t):  # we require alphabetic
            continue  # we continue
        if len(t) < 2:  # we skip single chars
            continue  # we continue
            
        # we categorize by position in vocab (rough frequency proxy)
        if i < max_words // 10:  # we top 10% = common
            common_words.append(t)  # we add common
        elif i < max_words // 3:  # we next 23% = medium
            medium_words.append(t)  # we add medium
        else:  # we rest = rare
            rare_words.append(t)  # we add rare
    
    # we ensure we have enough words in each category
    if len(common_words) < 500:
        extra = [tok for tok, _ in vocab_items if tok not in tokenizer.all_special_tokens][:1000]  # we fallback
        common_words.extend(extra)  # we extend
    if len(medium_words) < 1000:
        extra = [tok for tok, _ in vocab_items[max_words//10:max_words//2] if tok not in tokenizer.all_special_tokens]  # we fallback
        medium_words.extend(extra)  # we extend
    if len(rare_words) < 1000:
        extra = [tok for tok, _ in vocab_items[max_words//2:] if tok not in tokenizer.all_special_tokens]  # we fallback
        rare_words.extend(extra)  # we extend
        
    return common_words, medium_words, rare_words  # we return categorized


def generate_diverse_sentences(tokenizer: AutoTokenizer, n_sentences: int, min_words: int, max_words: int, seed: int) -> Tuple[List[str], List[Dict]]:
    """we generate diverse sentences with varying lengths and word types for better dot product distribution"""
    rng = random.Random(seed)  # we create rng
    common, medium, rare = filter_vocab_words(tokenizer)  # we get categorized vocab
    
    sents: List[str] = []  # we init sentences
    sent_info: List[Dict] = []  # we detailed info per sentence
    
    for i in range(n_sentences):
        # we vary sentence length
        n_words = rng.randint(min_words, max_words)  # we random length
        
        # we mix word types for diversity
        n_common = max(1, n_words // 3)  # we at least 1/3 common
        n_medium = max(1, n_words // 3)  # we at least 1/3 medium  
        n_rare = n_words - n_common - n_medium  # we rest rare
        
        chosen_words = []  # we word list
        word_types = []  # we track types
        
        # we sample from each category
        if len(common) >= n_common:
            words_c = rng.sample(common, k=n_common)  # we sample common
            chosen_words.extend(words_c)  # we add
            word_types.extend(["common"] * n_common)  # we track
            
        if len(medium) >= n_medium:
            words_m = rng.sample(medium, k=n_medium)  # we sample medium
            chosen_words.extend(words_m)  # we add
            word_types.extend(["medium"] * n_medium)  # we track
            
        if n_rare > 0 and len(rare) >= n_rare:
            words_r = rng.sample(rare, k=n_rare)  # we sample rare
            chosen_words.extend(words_r)  # we add
            word_types.extend(["rare"] * n_rare)  # we track
        
        # we shuffle order
        combined = list(zip(chosen_words, word_types))  # we zip
        rng.shuffle(combined)  # we shuffle
        chosen_words, word_types = zip(*combined)  # we unzip
        
        sent = " ".join(chosen_words)  # we build sentence
        sents.append(sent)  # we store
        
        # we detailed info
        info = {
            "sentence_idx": i,
            "sentence": sent,
            "words": list(chosen_words),
            "word_types": list(word_types),
            "n_words": len(chosen_words),
            "n_common": word_types.count("common"),
            "n_medium": word_types.count("medium"),
            "n_rare": word_types.count("rare"),
        }  # we info dict
        sent_info.append(info)  # we store info
        
    return sents, sent_info  # we return sentences and info


def analyze_dot_product_coverage(dots: np.ndarray, min_bucket_size: float = 0.1) -> Dict:
    """we analyze how well dot products cover the [-1, 1] range with given bucket size"""
    # we extract upper triangular pairs
    N = dots.shape[0]  # we size
    iu, ju = np.triu_indices(N, k=1)  # we upper triangle
    dot_pairs = dots[iu, ju]  # we extract pairs
    
    # we analyze range and distribution
    dot_min, dot_max = float(np.min(dot_pairs)), float(np.max(dot_pairs))  # we range
    n_buckets = int(2.0 / min_bucket_size)  # we number of buckets needed
    bucket_edges = np.linspace(-1.0, 1.0, n_buckets + 1)  # we bucket edges
    
    # we count pairs per bucket
    bucket_counts, _ = np.histogram(dot_pairs, bins=bucket_edges)  # we histogram
    empty_buckets = int(np.sum(bucket_counts == 0))  # we count empty buckets
    coverage = (n_buckets - empty_buckets) / n_buckets  # we coverage fraction
    
    analysis = {
        "dot_min": dot_min,
        "dot_max": dot_max,
        "range_covered": dot_max - dot_min,
        "n_pairs": len(dot_pairs),
        "n_buckets": n_buckets,
        "empty_buckets": empty_buckets,
        "coverage_fraction": coverage,
        "bucket_counts": bucket_counts.tolist(),
        "bucket_edges": bucket_edges.tolist(),
        "min_bucket_size": min_bucket_size,
    }  # we analysis dict
    
    return analysis  # we return analysis


def generate_prescribed_vectors_optimal(n_sentences: int, min_bucket_size: float, d_embed: int, seed: int) -> Tuple[List[np.ndarray], List[float]]:
    """we generate vectors to optimally cover dot product buckets for smooth curves"""
    rng = np.random.RandomState(seed)  # we create rng
    
    # we calculate available pairs from n_sentences
    n_pairs = (n_sentences * (n_sentences - 1)) // 2  # we upper triangular pairs
    n_buckets = int(2.0 / min_bucket_size)  # we total buckets needed (20 for 0.1)
    
    print(f"    generating {n_sentences} vectors for {n_pairs} pairs to cover {n_buckets} buckets")  # we report
    
    # we always aim for dense uniform coverage for smooth curves
    target_dots = np.linspace(-0.95, 0.95, n_pairs).tolist()  # we dense uniform coverage
    
    # we shuffle to avoid systematic patterns in optimization
    rng.shuffle(target_dots)  # we randomize order for better convergence
    
    # we use constraint satisfaction to generate vectors with exact pairwise dot products
    vectors = generate_constrained_vectors(n_sentences, target_dots, d_embed, rng)  # we constrained generation
    
    return vectors, target_dots  # we return vectors and target dots


def generate_constrained_vectors(n_vectors: int, target_dots: List[float], d_embed: int, rng: np.random.RandomState) -> List[np.ndarray]:
    """we generate n_vectors with prescribed pairwise dot products using optimization"""
    
    # we start with random unit vectors
    vectors = []  # we vector list
    for i in range(n_vectors):
        v = rng.randn(d_embed)  # we random vector
        v = v / np.linalg.norm(v)  # we normalize
        vectors.append(v)  # we store
    
    # we iteratively adjust vectors to match target dot products
    n_iterations = 2000  # we increased iteration limit for convergence
    learning_rate = 0.005  # we smaller step size for stability
    
    for iteration in range(n_iterations):
        total_error = 0.0  # we error accumulator
        
        # we process each pair
        pair_idx = 0  # we pair counter
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                if pair_idx < len(target_dots):
                    target_dot = target_dots[pair_idx]  # we target for this pair
                    current_dot = np.dot(vectors[i], vectors[j])  # we current dot
                    error = target_dot - current_dot  # we error
                    total_error += error**2  # we accumulate squared error
                    
                    # we gradient update to reduce error
                    grad_i = -error * vectors[j]  # we gradient for vector i
                    grad_j = -error * vectors[i]  # we gradient for vector j
                    
                    # we update vectors with momentum-like adjustment
                    vectors[i] = vectors[i] + learning_rate * grad_i  # we update i
                    vectors[j] = vectors[j] + learning_rate * grad_j  # we update j
                    
                    # we renormalize to unit length
                    vectors[i] = vectors[i] / np.linalg.norm(vectors[i])  # we normalize i
                    vectors[j] = vectors[j] / np.linalg.norm(vectors[j])  # we normalize j
                
                pair_idx += 1  # we increment pair counter
        
        # we check convergence
        if total_error < 1e-5:  # we relaxed convergence for large systems
            print(f"    converged after {iteration+1} iterations, error={total_error:.2e}")  # we report
            break  # we converged
        
        # we adaptive learning rate decay
        if iteration % 200 == 199:
            learning_rate *= 0.95  # we gradual decay
        
        # we progress report for large systems
        if iteration % 500 == 499:
            print(f"    iteration {iteration+1}, error={total_error:.2e}")  # we progress
    
    if iteration == n_iterations - 1:
        print(f"    reached max iterations, final error={total_error:.2e}")  # we report
    
    return vectors  # we return optimized vectors


def find_closest_tokens(
    target_vectors: List[np.ndarray],
    emb_matrix: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 5,
    allowed_ids: np.ndarray | None = None,
) -> List[Dict]:
    """we find tokens with embeddings closest to target vectors, restricted to whole-word tokens"""
    emb_np = emb_matrix.detach().cpu().numpy()  # we move to numpy
    emb_normalized = emb_np / np.linalg.norm(emb_np, axis=1, keepdims=True)  # we normalize embeddings

    if allowed_ids is None:  # we compute allowed ids if not provided
        allowed_ids = get_allowed_token_ids(tokenizer, emb_np.shape[0])  # we allowed ids

    # we pre-extract allowed embeddings
    allowed_emb = emb_normalized[allowed_ids]  # we allowed rows

    results: List[Dict] = []  # we store results

    for i, target_vec in enumerate(target_vectors):
        # we compute cosine similarities restricted to allowed tokens
        similarities = np.dot(allowed_emb, target_vec)  # we dot products

        # we find top k matches among allowed ids
        top_local = np.argsort(similarities)[-k:][::-1]  # we sort descending
        top_ids = allowed_ids[top_local]  # we map back to global ids
        top_sims = similarities[top_local]  # we sims

        # we get tokens
        tokens = []  # we token list
        for idx, sim in zip(top_ids, top_sims):
            token = tokenizer.convert_ids_to_tokens([int(idx)])[0]  # we get token
            tokens.append({"token": token, "id": int(idx), "similarity": float(sim)})  # we store

        result = {
            "vector_idx": i,
            "target_norm": float(np.linalg.norm(target_vec)),
            "closest_tokens": tokens,
        }  # we result dict
        results.append(result)  # we store

    return results  # we return results


def create_sentences_from_optimal_vectors(
    target_vectors: List[np.ndarray], 
    target_dots: List[float],
    emb_matrix: torch.Tensor, 
    tokenizer: AutoTokenizer,
    n_encoder_sentences: int = 2,
    n_decoder_sentences: int = 2,
) -> Tuple[List[str], List[Dict], np.ndarray]:
    """we create sentences from optimally distributed vectors ensuring maximum dot product coverage"""
    total_sentences = n_encoder_sentences + n_decoder_sentences  # we total sentences needed
    
    if len(target_vectors) < total_sentences:
        raise ValueError(f"need {total_sentences} vectors but only got {len(target_vectors)}")  # we check
    
    # we find closest tokens among allowed whole-word tokens
    allowed_ids = get_allowed_token_ids(tokenizer, emb_matrix.shape[0])  # we allowed ids
    token_matches = find_closest_tokens(
        target_vectors[:total_sentences], emb_matrix, tokenizer, k=1, allowed_ids=allowed_ids
    )  # we find matches
    
    # we build sentences (1 token each for simplicity)
    sentences: List[str] = []  # we sentences
    sent_info: List[Dict] = []  # we info
    selected_vectors: List[np.ndarray] = []  # we actual vectors used
    
    for i in range(total_sentences):
        best_token = token_matches[i]["closest_tokens"][0]  # we best match
        sentence = best_token["token"]  # we single token sentence
        
        # we use the optimized target vector directly to maintain exact dot products
        prescribed_vector = target_vectors[i].copy()  # we use prescribed vector
        selected_vectors.append(prescribed_vector)  # we store prescribed
        
        sentences.append(sentence)  # we store sentence
        
        # we compute which target dot this corresponds to
        pair_idx = 0  # we pair index
        relevant_dots = []  # we relevant target dots for this vector
        for ii in range(total_sentences):
            for jj in range(ii + 1, total_sentences):
                if (ii == i or jj == i) and pair_idx < len(target_dots):
                    relevant_dots.append(target_dots[pair_idx])  # we add relevant dot
                pair_idx += 1  # we increment
        
        info = {
            "sentence_idx": i,
            "sentence": sentence,
            "words": [sentence],  # we single word
            "word_types": ["prescribed_optimal"],  # we type
            "n_words": 1,
            "target_vector_idx": i,
            "relevant_target_dots": relevant_dots,  # we store all relevant dots
            "token_id": best_token["id"],
            "similarity_to_target": best_token["similarity"],
            "sentence_type": "encoder" if i < n_encoder_sentences else "decoder",
            "uses_prescribed_vector": True,  # we flag prescribed usage
        }  # we info
        sent_info.append(info)  # we store
    
    # we compute dot products matrix using prescribed vectors
    vectors_matrix = np.stack(selected_vectors, axis=0)  # we stack [N, d]
    dots = np.dot(vectors_matrix, vectors_matrix.T).astype(np.float32)  # we dot products
    
    # we verify we achieved target dot products
    pair_idx = 0  # we pair counter
    max_error = 0.0  # we max error tracker
    for i in range(total_sentences):
        for j in range(i + 1, total_sentences):
            if pair_idx < len(target_dots):
                target = target_dots[pair_idx]  # we target
                actual = dots[i, j]  # we actual
                error = abs(target - actual)  # we error
                max_error = max(max_error, error)  # we track max error
            pair_idx += 1  # we increment
    
    print(f"    max dot product error: {max_error:.4f}")  # we report error
    
    return sentences, sent_info, dots  # we return


def sentence_token_ids(tokenizer: AutoTokenizer, sentence: str) -> Tuple[List[int], Dict]:
    """we tokenize a sentence and return ids plus tokenization info"""
    pieces = sentence.split()  # we split by spaces
    ids: List[int] = []  # we init list
    subword_counts: List[int] = []  # we track subwords per word
    
    for w in pieces:
        toks = tokenizer.tokenize(w)  # we tokenize word
        if len(toks) == 0:
            toks = [tokenizer.unk_token]  # we fallback unk
        tok_ids = tokenizer.convert_tokens_to_ids(toks)  # we convert to ids
        ids.append(int(tok_ids[0]))  # we keep first subword
        subword_counts.append(len(toks))  # we count subwords
    
    # we tokenization info
    token_info = {
        "n_words": len(pieces),
        "n_tokens": len(ids),
        "subword_counts": subword_counts,
        "avg_subwords_per_word": float(np.mean(subword_counts)) if subword_counts else 0.0,
        "max_subwords_per_word": int(max(subword_counts)) if subword_counts else 0,
    }  # we info dict
    
    return ids, token_info  # we return ids and info


def embed_and_normalize_tokens(token_ids: List[int], emb_matrix: torch.Tensor) -> torch.Tensor:
    """we fetch embeddings for token ids and normalize each token to unit norm"""
    idx = torch.tensor(token_ids, dtype=torch.long, device=emb_matrix.device)  # we build index tensor
    X = emb_matrix.index_select(0, idx)  # we select embeddings [T, d_embed]
    X = F.normalize(X, p=2.0, dim=-1)  # we normalize per token
    return X  # we return normalized tokens


def sentence_mean_unit_vector(X: torch.Tensor) -> torch.Tensor:
    """we average token vectors, then normalize to unit length"""
    m = X.mean(dim=0, keepdim=False)  # we average tokens
    m = F.normalize(m, p=2.0, dim=-1)  # we normalize mean
    return m  # we return unit vector


class EncoderDecoderAttention(nn.Module):
    """we implement a simple encoder-decoder attention architecture with controlled token counts"""

    def __init__(self, d_embed: int, d_model: int, n_heads: int, n_encoder_tokens: int = 1, n_decoder_tokens: int = 1):
        super().__init__()  # we init parent
        assert d_model % n_heads == 0, "we require d_model divisible by n_heads"  # we check divisibility
        self.d_embed = d_embed  # we store
        self.d_model = d_model  # we store
        self.n_heads = n_heads  # we store
        self.d_head = d_model // n_heads  # we compute head dim
        self.n_encoder_tokens = n_encoder_tokens  # we encoder length
        self.n_decoder_tokens = n_decoder_tokens  # we decoder length

        # we encoder components
        self.encoder_proj = nn.Linear(d_embed, d_model, bias=False)  # we project encoder input
        self.encoder_W_q = nn.Linear(d_model, d_model, bias=False)  # we encoder queries
        self.encoder_W_k = nn.Linear(d_model, d_model, bias=False)  # we encoder keys
        self.encoder_W_v = nn.Linear(d_model, d_model, bias=False)  # we encoder values
        self.encoder_W_o = nn.Linear(d_model, d_model, bias=False)  # we encoder output

        # we decoder components
        self.decoder_proj = nn.Linear(d_embed, d_model, bias=False)  # we project decoder input
        self.decoder_W_q = nn.Linear(d_model, d_model, bias=False)  # we decoder queries
        self.decoder_W_k = nn.Linear(d_model, d_model, bias=False)  # we decoder keys (for self-attn)
        self.decoder_W_v = nn.Linear(d_model, d_model, bias=False)  # we decoder values (for self-attn)
        self.decoder_W_o = nn.Linear(d_model, d_model, bias=False)  # we decoder self-attn output

        # we cross-attention components
        self.cross_W_q = nn.Linear(d_model, d_model, bias=False)  # we cross queries (from decoder)
        self.cross_W_k = nn.Linear(d_model, d_model, bias=False)  # we cross keys (from encoder)
        self.cross_W_v = nn.Linear(d_model, d_model, bias=False)  # we cross values (from encoder)
        self.cross_W_o = nn.Linear(d_model, d_model, bias=False)  # we cross output

        self.readout = nn.Linear(d_model, 1, bias=False)  # we scalar readout

        self.reset_parameters()  # we init weights

    def reset_parameters(self) -> None:
        """we use fan-in scaling to keep gradients stable"""
        all_linears = [
            self.encoder_proj, self.encoder_W_q, self.encoder_W_k, self.encoder_W_v, self.encoder_W_o,
            self.decoder_proj, self.decoder_W_q, self.decoder_W_k, self.decoder_W_v, self.decoder_W_o,
            self.cross_W_q, self.cross_W_k, self.cross_W_v, self.cross_W_o
        ]  # we all linear layers
        for m in all_linears:
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear", mode="fan_in")  # we init linear weights
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / np.sqrt(self.d_model))  # we init readout

    def attention_block(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, W_o: nn.Linear) -> torch.Tensor:
        """we compute multi-head attention block"""
        T_q, T_kv = Q.shape[0], K.shape[0]  # we sequence lengths
        H, Dh = self.n_heads, self.d_head  # we heads and head dim

        def split_heads_q(Z: torch.Tensor) -> torch.Tensor:
            return Z.view(T_q, H, Dh).transpose(0, 1).contiguous()  # we [H, T_q, Dh]
        
        def split_heads_kv(Z: torch.Tensor) -> torch.Tensor:
            return Z.view(T_kv, H, Dh).transpose(0, 1).contiguous()  # we [H, T_kv, Dh]

        Qh, Kh, Vh = split_heads_q(Q), split_heads_kv(K), split_heads_kv(V)  # we split heads

        attn_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) / np.sqrt(Dh)  # we compute logits [H, T_q, T_kv]
        attn_weights = F.softmax(attn_logits, dim=-1)  # we softmax over keys [H, T_q, T_kv]
        context = torch.matmul(attn_weights, Vh)  # we apply attention [H, T_q, Dh]

        context = context.transpose(0, 1).contiguous().view(T_q, H * Dh)  # we merge heads [T_q, d_model]
        out = W_o(context)  # we project out [T_q, d_model]
        return out  # we return output

    def forward(self, X_tokens: torch.Tensor) -> torch.Tensor:
        """we forward encoder-decoder with controlled token counts"""
        T = X_tokens.shape[0]  # we total sequence length
        
        # we split into encoder/decoder parts (for experiment: 1 encoder + 1 decoder)
        enc_len = min(self.n_encoder_tokens, T)  # we encoder length
        dec_len = min(self.n_decoder_tokens, T - enc_len)  # we decoder length
        
        if enc_len == 0:  # we handle edge case
            enc_len = 1  # we at least 1 encoder token
            dec_len = min(self.n_decoder_tokens, T - 1)  # we adjust decoder
        if dec_len == 0:  # we handle edge case
            dec_len = 1  # we at least 1 decoder token
            
        X_enc = X_tokens[:enc_len]  # we encoder tokens [enc_len, d_embed]
        X_dec = X_tokens[enc_len:enc_len+dec_len] if enc_len + dec_len <= T else X_tokens[-dec_len:]  # we decoder tokens

        # we encoder self-attention
        H_enc = self.encoder_proj(X_enc)  # we project encoder [enc_len, d_model]
        Q_enc = self.encoder_W_q(H_enc)  # we encoder queries
        K_enc = self.encoder_W_k(H_enc)  # we encoder keys
        V_enc = self.encoder_W_v(H_enc)  # we encoder values
        H_enc_out = self.attention_block(Q_enc, K_enc, V_enc, self.encoder_W_o)  # we encoder self-attention

        # we decoder self-attention
        H_dec = self.decoder_proj(X_dec)  # we project decoder [dec_len, d_model]
        Q_dec = self.decoder_W_q(H_dec)  # we decoder queries
        K_dec = self.decoder_W_k(H_dec)  # we decoder keys
        V_dec = self.decoder_W_v(H_dec)  # we decoder values
        H_dec_self = self.attention_block(Q_dec, K_dec, V_dec, self.decoder_W_o)  # we decoder self-attention

        # we cross-attention (decoder attends to encoder)
        Q_cross = self.cross_W_q(H_dec_self)  # we cross queries from decoder
        K_cross = self.cross_W_k(H_enc_out)  # we cross keys from encoder
        V_cross = self.cross_W_v(H_enc_out)  # we cross values from encoder
        H_dec_cross = self.attention_block(Q_cross, K_cross, V_cross, self.cross_W_o)  # we cross-attention

        # we final readout (mean pool decoder output)
        h = H_dec_cross.mean(dim=0, keepdim=False)  # we mean pool [d_model]
        y = self.readout(h)  # we read scalar [1]
        return y.squeeze(0)  # we return scalar

    def get_token_counts(self, sequence_length: int) -> Dict[str, int]:
        """we report actual token counts used by the model"""
        enc_len = min(self.n_encoder_tokens, sequence_length)  # we encoder length
        dec_len = min(self.n_decoder_tokens, sequence_length - enc_len)  # we decoder length
        
        if enc_len == 0:
            enc_len = 1
            dec_len = min(self.n_decoder_tokens, sequence_length - 1)
        if dec_len == 0:
            dec_len = 1
            
        return {
            "encoder_tokens": enc_len,
            "decoder_tokens": dec_len,
            "total_tokens": enc_len + dec_len,
            "sequence_length": sequence_length,
        }  # we return counts


class FCNNBaseline(nn.Module):
    """we implement a 2-layer mlp baseline mapping sequence to scalar"""

    def __init__(self, d_embed: int, d_model: int):
        super().__init__()  # we init parent
        self.proj_in = nn.Linear(d_embed, d_model, bias=False)  # we project input tokens
        self.lin1 = nn.Linear(d_model, d_model, bias=False)  # we first layer
        self.lin2 = nn.Linear(d_model, 1, bias=False)  # we second to scalar
        self.reset_parameters()  # we init params

    def reset_parameters(self) -> None:
        """we use kaiming for stable grads"""
        for m in [self.proj_in, self.lin1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu", mode="fan_in")  # we init with kaiming
        nn.init.normal_(self.lin2.weight, mean=0.0, std=1.0)  # we init output

    def forward(self, X_tokens: torch.Tensor) -> torch.Tensor:
        """we forward [T, d_embed] → scalar"""
        H_in = self.proj_in(X_tokens)  # we project tokens
        h = H_in.mean(dim=0, keepdim=False)  # we mean pool tokens
        h = F.relu(self.lin1(h))  # we apply relu
        y = self.lin2(h)  # we read scalar
        return y.squeeze(0)  # we return scalar


def flatten_grad(grad_list: List[torch.Tensor]) -> torch.Tensor:
    """we flatten grad tensors to a single 1d vector on cpu"""
    flat = torch.cat([g.reshape(-1) for g in grad_list if g is not None], dim=0)  # we concat grads
    return flat.detach().cpu()  # we return cpu vector


def grad_wrt_params(model: nn.Module, X_tokens: torch.Tensor) -> torch.Tensor:
    """we compute gradient of scalar output wrt parameters for one input sequence"""
    params = [p for p in model.parameters()]  # we get params
    for p in params:
        p.requires_grad_(True)  # we ensure grads
    y = model(X_tokens)  # we forward
    grads = torch.autograd.grad(y, params, retain_graph=False, create_graph=False, allow_unused=False)  # we grads
    g = flatten_grad(list(grads))  # we flatten
    return g  # we return grad vector


def build_memmap_matrix(path: Path, shape: Tuple[int, int], dtype: np.dtype = np.float32) -> np.memmap:
    """we create a memmap matrix file for incremental writes"""
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)  # we open memmap


def compute_ntk_incremental(
    model: nn.Module,
    token_seqs: List[torch.Tensor],
    device: torch.device,
    K_path: Path,
    csv_path: Path,
    label: str,
) -> np.memmap:
    """we compute NTK by accumulating gradients and writing K incrementally"""
    N = len(token_seqs)  # we number of inputs
    K = build_memmap_matrix(K_path, (N, N), dtype=np.float32)  # we allocate memmap
    G: List[torch.Tensor] = []  # we store gradients
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("label,i,j,K_ij\n")  # we write header
    for i in tqdm(range(N), desc=f"grads[{label}]", leave=False):
        g_i = grad_wrt_params(model, token_seqs[i].to(device))  # we compute gradient for i
        G.append(g_i)  # we store gradient
        gi_np = g_i.numpy()  # we to numpy
        K[i, i] = float(np.dot(gi_np, gi_np))  # we set diagonal
        if i > 0:
            Gi = gi_np  # we alias
            for j in range(i):
                Kj = float(np.dot(G[j].numpy(), Gi))  # we compute dot
                K[i, j] = Kj  # we set lower tri
                K[j, i] = Kj  # we mirror
                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{label},{i},{j},{Kj}\n")  # we append row
        K.flush()  # we flush memmap
    return K  # we return memmap


def upper_triangle_pairs(dots: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """we extract upper-triangular pairs (x, y) with x=dot, y=kernel"""
    N = K.shape[0]  # we size
    iu, ju = np.triu_indices(N, k=1)  # we upper tri indices
    x = dots[iu, ju].astype(np.float32)  # we extract dots
    y = K[iu, ju].astype(np.float32)  # we extract kernel
    return x, y  # we return pairs


def scatter_with_binning(ax, x: np.ndarray, y: np.ndarray, color: str, name: str, bins: int = 20) -> None:
    """we draw scatter and binned means"""
    ax.scatter(x, y, s=6, alpha=0.20, color=color, label=f"{name} (pairs)")  # we scatter raw pairs
    if len(x) == 0:
        return  # we handle empty
    edges = np.linspace(-1.0, 1.0, bins + 1)  # we make bin edges
    idx = np.digitize(x, edges) - 1  # we bin x
    means_x = []  # we init
    means_y = []  # we init
    stds_y = []  # we init
    counts = []  # we init
    for b in range(bins):
        sel = (idx == b)  # we select bin
        if np.any(sel):
            means_x.append(np.mean(x[sel]))  # we mean x
            means_y.append(np.mean(y[sel]))  # we mean y
            stds_y.append(np.std(y[sel]))  # we std y
            counts.append(int(np.sum(sel)))  # we count
    if len(means_x) > 0:
        ax.plot(means_x, means_y, color=color, linewidth=2.0, label=f"{name} (binned)")  # we plot binned curve
    return  # we return


def plot_attention_only(attn_series: List[Tuple[np.ndarray, np.ndarray, str]], out_path: Path, title: str, bins: int):
    """we plot attention-only curves, possibly multiple trials"""
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2"]  # we define colors
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0), dpi=140)  # we create figure
    for k, (x, y, name) in enumerate(attn_series):
        scatter_with_binning(ax, x, y, color=colors[k % len(colors)], name=name, bins=bins)  # we plot series
    ax.set_xlabel("dot(x, y) on unit sphere")  # we label x
    ax.set_ylabel("NTK(x, y) [attention]")  # we label y
    ax.set_title(title)  # we set title
    ax.grid(True, alpha=0.25)  # we grid
    ax.legend()  # we legend
    fig.tight_layout()  # we layout
    fig.savefig(out_path)  # we save fig
    plt.close(fig)  # we close fig


def save_binned_stats(x: np.ndarray, y: np.ndarray, bins: int, out_csv: Path) -> None:
    """we compute and save binned mean/std/count as csv"""
    edges = np.linspace(-1.0, 1.0, bins + 1)  # we edges
    idx = np.digitize(x, edges) - 1  # we bin
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("bin_center,mean,std,count\n")  # we header
        for b in range(bins):
            sel = (idx == b)  # we select bin
            if np.any(sel):
                xc = float(np.mean(x[sel]))  # we center
                mu = float(np.mean(y[sel]))  # we mean
                sd = float(np.std(y[sel]))  # we std
                ct = int(np.sum(sel))  # we count
                f.write(f"{xc},{mu},{sd},{ct}\n")  # we write row
    return  # we return


def plot_all_configs_consolidated(
    configs_data: List[Dict], 
    out_path: Path, 
    title: str, 
    bins: int = 20
) -> None:
    """we plot all configs on one figure for comparison"""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
             "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1a55FF", "#ff1a8c"]  # we define colors
    
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.0), dpi=140)  # we create larger figure
    
    for k, config in enumerate(configs_data):
        x = config["x_data"]  # we get x
        y = config["y_data"]  # we get y
        label = config["label"]  # we get label
        color = colors[k % len(colors)]  # we cycle colors
        
        # we plot binned curve only (no scatter for clarity)
        if len(x) > 0:
            edges = np.linspace(-1.0, 1.0, bins + 1)  # we make bin edges
            idx = np.digitize(x, edges) - 1  # we bin x
            means_x = []  # we init
            means_y = []  # we init
            for b in range(bins):
                sel = (idx == b)  # we select bin
                if np.any(sel):
                    means_x.append(np.mean(x[sel]))  # we mean x
                    means_y.append(np.mean(y[sel]))  # we mean y
            if len(means_x) > 0:
                ax.plot(means_x, means_y, color=color, linewidth=2.5, label=label, marker="o", markersize=4)  # we plot
    
    ax.set_xlabel("dot(x, y) on unit sphere")  # we label x
    ax.set_ylabel("NTK(x, y) [attention]")  # we label y
    ax.set_title(title)  # we set title
    ax.grid(True, alpha=0.3)  # we grid
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # we legend outside
    fig.tight_layout()  # we layout
    fig.savefig(out_path, bbox_inches="tight")  # we save with bbox
    plt.close(fig)  # we close


def save_config_json(config_dict: Dict, out_path: Path) -> None:
    """we save configuration as json"""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)  # we save json


def main():
    parser = argparse.ArgumentParser(description="we compute empirical ntk for encoder-decoder attention with controlled dot product distribution")  # we parser
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="we choose hf model name")  # we arg
    parser.add_argument("--min_bucket_size", type=float, default=0.1, help="we minimum bucket size for dot product coverage")  # we arg
    parser.add_argument("--coverage_threshold", type=float, default=0.95, help="we minimum coverage fraction to use natural sentences")  # we arg
    parser.add_argument("--n_encoder_sentences", type=int, default=10, help="we number of encoder sentences")  # we arg
    parser.add_argument("--n_decoder_sentences", type=int, default=10, help="we number of decoder sentences")  # we arg
    parser.add_argument("--heads", type=str, default="1,2,4,8", help="we comma list of head counts")  # we arg
    parser.add_argument("--dims", type=str, default="64,128,256", help="we comma list of internal d_model")  # we arg
    parser.add_argument("--trials", type=int, default=3, help="we number of trials with new sentences")  # we arg
    parser.add_argument("--aggregate_bins", type=int, default=20, help="we number of bins for aggregation")  # we arg
    parser.add_argument("--seed", type=int, default=123, help="we random seed")  # we arg
    parser.add_argument("--device", type=str, default="cuda", help="we device cuda or cpu")  # we arg
    parser.add_argument("--name", type=str, default="orion_ntk", help="we run name used in filenames")  # we arg
    parser.add_argument("--consolidate_sentence_idx", type=int, default=0, help="we sentence index for consolidated plot")  # we arg
    parser.add_argument("--n_encoder_tokens", type=int, default=1, help="we tokens used in encoder")  # we arg
    parser.add_argument("--n_decoder_tokens", type=int, default=1, help="we tokens used in decoder")  # we arg
    args = parser.parse_args()  # we parse

    set_seed(args.seed)  # we set seed
    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")  # we pick device
    this_file = Path(__file__)  # we locate this file
    plots_dir, data_dir = resolve_paths(this_file)  # we resolve output dirs
    tokenizer, emb_matrix = load_hf_embeddings(args.model_name, dev)  # we load tokenizer and embeddings

    heads_list = [int(x.strip()) for x in args.heads.split(",") if x.strip()]  # we parse heads
    dims_list = [int(x.strip()) for x in args.dims.split(",") if x.strip()]  # we parse dims
    d_embed = emb_matrix.shape[1]  # we get embedding dimension

    # we save main config json
    main_config = {
        "model_name": args.model_name,
        "d_embed": int(d_embed),
        "min_bucket_size": args.min_bucket_size,
        "coverage_threshold": args.coverage_threshold,
        "n_encoder_sentences": args.n_encoder_sentences,
        "n_decoder_sentences": args.n_decoder_sentences,
        "heads_list": heads_list,
        "dims_list": dims_list,
        "trials": args.trials,
        "aggregate_bins": args.aggregate_bins,
        "seed": args.seed,
        "device": args.device,
        "name": args.name,
        "consolidate_sentence_idx": args.consolidate_sentence_idx,
        "n_encoder_tokens": args.n_encoder_tokens,
        "n_decoder_tokens": args.n_decoder_tokens,
        "architecture": "encoder_decoder_attention",
        "dot_product_strategy": "controlled_coverage",
    }  # we config dict
    config_json_path = data_dir / f"{args.name}_main_config.json"  # we path
    save_config_json(main_config, config_json_path)  # we save main config

    # we calculate and report expected coverage with current settings
    n_encoder = args.n_encoder_sentences  # we encoder count
    n_decoder = args.n_decoder_sentences  # we decoder count
    total_sentences = n_encoder + n_decoder  # we total sentences
    n_pairs = (total_sentences * (total_sentences - 1)) // 2  # we total pairs
    n_buckets = int(2.0 / args.min_bucket_size)  # we buckets needed
    coverage_pct = (n_pairs / n_buckets) * 100  # we coverage percentage
    points_per_bucket = n_pairs / n_buckets  # we points per bucket
    
    print(f"=== DOT PRODUCT COVERAGE ANALYSIS ===")  # we header
    print(f"Avec {n_encoder} phrases encoder + {n_decoder} phrases decoder = {total_sentences} phrases total")  # we total
    print(f"Nombre de paires = C({total_sentences},2) = {n_pairs} points")  # we pairs
    print(f"Pour buckets de {args.min_bucket_size} sur [-1,1] = {n_buckets} buckets")  # we buckets
    print(f"Couverture = {n_pairs}/{n_buckets} = {coverage_pct:.0f}%")  # we coverage
    print(f"Points par bucket en moyenne = {points_per_bucket:.1f}")  # we density
    print(f"=====================================")  # we footer

    # we prepare consolidation data for specific sentence
    consolidate_data: List[Dict] = []  # we collect configs for consolidated plot

    for d_model in dims_list:
        for n_heads in heads_list:
            if d_model % n_heads != 0:
                continue  # we skip invalid combos
            config_id = f"h{n_heads}_d{d_model}"  # we id config
            print(f"[config] {config_id}")  # we log

            # we instantiate fresh models per config
            attn = EncoderDecoderAttention(
                d_embed=d_embed, 
                d_model=d_model, 
                n_heads=n_heads, 
                n_encoder_tokens=args.n_encoder_tokens,
                n_decoder_tokens=args.n_decoder_tokens
            ).to(dev)  # we make encoder-decoder attn
            fcnn = FCNNBaseline(d_embed=d_embed, d_model=d_model).to(dev)  # we make fcnn
            attn.eval(), fcnn.eval()  # we eval

            # we containers for aggregation
            agg_attn_x: List[np.ndarray] = []  # we x arrays
            agg_attn_y: List[np.ndarray] = []  # we y arrays

            for t in tqdm(range(args.trials), desc=f"trials[{config_id}]"):
                trial_seed = args.seed + 1000 * (t + 1)  # we set trial seed
                
                # we first try natural sentences to check coverage
                print(f"    trial {t}: analyzing dot product coverage...")  # we report
                
                # we generate a test set of diverse sentences to analyze coverage
                test_sentences, test_info = generate_diverse_sentences(
                    tokenizer, 50, 3, 12, seed=trial_seed + 123
                )  # we gen test set
                
                # we get test embeddings and dot products
                test_vectors = []  # we test vectors
                for s in test_sentences[:20]:  # we limit to 20 for analysis
                    ids, _ = sentence_token_ids(tokenizer, s)  # we tokenize
                    X = embed_and_normalize_tokens(ids, emb_matrix)  # we embed
                    test_vectors.append(sentence_mean_unit_vector(X))  # we mean vector
                
                with torch.no_grad():
                    test_S = torch.stack(test_vectors, dim=0)  # we stack
                    test_S = F.normalize(test_S, p=2.0, dim=-1)  # we normalize
                    test_dots = (test_S @ test_S.t()).detach().cpu().numpy().astype(np.float32)  # we test dots
                
                # we analyze coverage
                coverage_analysis = analyze_dot_product_coverage(test_dots, args.min_bucket_size)  # we analyze
                
                use_prescribed = coverage_analysis["coverage_fraction"] < args.coverage_threshold  # we decide
                
                if use_prescribed:
                    print(f"    trial {t}: coverage {coverage_analysis['coverage_fraction']:.2f} < {args.coverage_threshold}, using optimal prescribed vectors")  # we report
                    
                    # we generate optimally distributed vectors for maximum coverage
                    total_sentences = args.n_encoder_sentences + args.n_decoder_sentences  # we total sentences
                    target_vectors, target_dots = generate_prescribed_vectors_optimal(
                        total_sentences, args.min_bucket_size, d_embed, trial_seed + 456
                    )  # we generate optimal vectors
                    
                    # we create sentences from optimal vectors
                    sentences, sent_info, dots = create_sentences_from_optimal_vectors(
                        target_vectors, target_dots, emb_matrix, tokenizer,
                        args.n_encoder_sentences, args.n_decoder_sentences
                    )  # we create from optimal vectors
                    
                    # we extract the selected vectors for later use
                    selected_vectors = target_vectors[:len(sentences)]  # we take needed vectors
                    
                    strategy = "optimal_prescribed_vectors"  # we strategy
                    
                else:
                    print(f"    trial {t}: coverage {coverage_analysis['coverage_fraction']:.2f} >= {args.coverage_threshold}, using natural sentences")  # we report
                    
                    # we use natural diverse sentences
                    sentences, sent_info = generate_diverse_sentences(
                        tokenizer, args.n_encoder_sentences + args.n_decoder_sentences, 3, 10, seed=trial_seed
                    )  # we gen natural
                    
                    # we mark encoder/decoder sentences
                    for i, info in enumerate(sent_info):
                        info["sentence_type"] = "encoder" if i < args.n_encoder_sentences else "decoder"  # we mark type
                    
                    selected_vectors = None  # we no prescribed vectors
                    strategy = "natural_sentences"  # we strategy

                # we embed and normalize tokens for each sentence
                token_seqs: List[torch.Tensor] = []  # we store token embeddings sequences
                sent_vectors: List[torch.Tensor] = []  # we store mean unit vectors
                token_infos: List[Dict] = []  # we store tokenization info
                
                for i, s in enumerate(tqdm(sentences, desc="embed+normalize", leave=False)):
                    ids, tok_info = sentence_token_ids(tokenizer, s)  # we get word token ids + info
                    X = embed_and_normalize_tokens(ids, emb_matrix)  # we get normalized embeddings [T,d_embed]
                    token_seqs.append(X)  # we store tokens
                    
                    # we check if using prescribed vectors
                    if (use_prescribed and selected_vectors is not None and 
                        i < len(sent_info) and sent_info[i].get("uses_prescribed_vector", False)):
                        # we use the prescribed vector instead of computing from embeddings
                        prescribed_vec = selected_vectors[i]  # we get prescribed from earlier
                        sent_vectors.append(torch.from_numpy(prescribed_vec).float())  # we convert to tensor
                    else:
                        # we compute mean vector from embeddings as usual
                        sent_vectors.append(sentence_mean_unit_vector(X))  # we store mean vector
                    
                    # we combine sentence info with tokenization info
                    if i < len(sent_info):
                        combined_info = {**sent_info[i], **tok_info}  # we merge dicts
                    else:
                        combined_info = tok_info  # we use tokenization info only
                        combined_info["sentence_type"] = "encoder" if i < args.n_encoder_sentences else "decoder"  # we set type
                    
                    # we add model-specific token counts
                    model_counts = attn.get_token_counts(len(ids))  # we get model token usage
                    combined_info.update(model_counts)  # we add model counts
                    combined_info["strategy"] = strategy  # we add strategy
                    token_infos.append(combined_info)  # we store combined info

                # we compute sentence dot products matrix (using prescribed vectors if available)
                if use_prescribed and 'dots' in locals():
                    pass  # we already have dots from prescribed vectors
                else:
                    with torch.no_grad():
                        S = torch.stack(sent_vectors, dim=0)  # we stack [N,d]
                        S = F.normalize(S, p=2.0, dim=-1)  # we ensure unit
                        dots = (S @ S.t()).detach().cpu().numpy().astype(np.float32)  # we compute pairwise dot products
                
                # we analyze final coverage
                final_analysis = analyze_dot_product_coverage(dots, args.min_bucket_size)  # we final analysis

                # we save sentences, detailed info and dot products for this trial
                sent_txt = data_dir / f"{args.name}_{config_id}_trial{t}_sentences.txt"  # we path text
                with open(sent_txt, "w", encoding="utf-8") as f:
                    for s in sentences:
                        f.write(s + "\n")  # we save sentences
                        
                # we save detailed sentence and tokenization info
                sent_info_json = data_dir / f"{args.name}_{config_id}_trial{t}_sentence_info.json"  # we path info
                save_config_json(token_infos, sent_info_json)  # we save detailed info
                
                dots_path = data_dir / f"{args.name}_{config_id}_trial{t}_dots.npy"  # we path dots
                np.save(dots_path, dots)  # we save dots
                
                # we save coverage analysis
                coverage_json = data_dir / f"{args.name}_{config_id}_trial{t}_coverage_analysis.json"  # we path coverage
                save_config_json({
                    "test_coverage": coverage_analysis,
                    "final_coverage": final_analysis,
                    "strategy_used": strategy,
                    "use_prescribed": use_prescribed,
                }, coverage_json)  # we save coverage analysis
                
                # we print comprehensive statistics
                enc_tokens = [info["encoder_tokens"] for info in token_infos]  # we encoder counts
                dec_tokens = [info["decoder_tokens"] for info in token_infos]  # we decoder counts
                seq_lengths = [info["sequence_length"] for info in token_infos]  # we sequence lengths
                encoder_sents = [info for info in token_infos if info.get("sentence_type") == "encoder"]  # we encoder sentences
                decoder_sents = [info for info in token_infos if info.get("sentence_type") == "decoder"]  # we decoder sentences
                
                print(f"    trial {t}: strategy={strategy}")  # we strategy
                print(f"    trial {t}: encoder_sents={len(encoder_sents)}, decoder_sents={len(decoder_sents)}")  # we counts
                print(f"    trial {t}: encoder_tokens={enc_tokens[0]}, decoder_tokens={dec_tokens[0]}")  # we tokens
                print(f"    trial {t}: final_coverage={final_analysis['coverage_fraction']:.2f}, range=[{final_analysis['dot_min']:.2f}, {final_analysis['dot_max']:.2f}]")  # we coverage

                # we kernel paths
                K_attn_path = data_dir / f"{args.name}_{config_id}_trial{t}_K_attn.dat"  # we path memmap
                K_fcnn_path = data_dir / f"{args.name}_{config_id}_trial{t}_K_fcnn.dat"  # we path memmap
                csv_attn_path = data_dir / f"{args.name}_{config_id}_trial{t}_pairs_attn.csv"  # we path csv
                csv_fcnn_path = data_dir / f"{args.name}_{config_id}_trial{t}_pairs_fcnn.csv"  # we path csv

                # we compute ntk incrementally for attention and fcnn
                K_attn = compute_ntk_incremental(attn, token_seqs, dev, K_attn_path, csv_attn_path, label="attn")  # we run
                K_fcnn = compute_ntk_incremental(fcnn, token_seqs, dev, K_fcnn_path, csv_fcnn_path, label="fcnn")  # we run

                # we also save .npy snapshots
                K_attn_npy = data_dir / f"{args.name}_{config_id}_trial{t}_K_attn.npy"  # we path npy
                K_fcnn_npy = data_dir / f"{args.name}_{config_id}_trial{t}_K_fcnn.npy"  # we path npy
                np.save(K_attn_npy, np.array(K_attn, copy=True))  # we save attn npy
                np.save(K_fcnn_npy, np.array(K_fcnn, copy=True))  # we save fcnn npy

                # we prepare attention-only pairs and per-trial plot
                x_attn, y_attn = upper_triangle_pairs(dots=dots, K=np.array(K_attn, copy=False))  # we pairs
                agg_attn_x.append(x_attn)  # we collect x
                agg_attn_y.append(y_attn)  # we collect y
                trial_plot = plots_dir / f"{args.name}_{config_id}_trial{t}_attn_K_vs_dot.png"  # we path fig
                plot_attention_only(
                    attn_series=[(x_attn, y_attn, f"attn trial {t}")],
                    out_path=trial_plot,
                    title=f"NTK vs dot (attention only) [{config_id}] trial {t}",
                    bins=args.aggregate_bins,
                )  # we plot attention only

                torch.cuda.empty_cache()  # we free cache

            # we aggregate trials for attention
            if len(agg_attn_x) > 0:
                X_all = np.concatenate(agg_attn_x, axis=0)  # we concat x
                Y_all = np.concatenate(agg_attn_y, axis=0)  # we concat y
                agg_plot = plots_dir / f"{args.name}_{config_id}_attn_K_vs_dot_aggregate.png"  # we path fig
                plot_attention_only(
                    attn_series=[(X_all, Y_all, "attention (all trials)")],
                    out_path=agg_plot,
                    title=f"NTK vs dot (attention only) aggregate [{config_id}]",
                    bins=args.aggregate_bins,
                )  # we plot aggregate
                agg_csv = data_dir / f"{args.name}_{config_id}_attn_binned_stats.csv"  # we path csv
                save_binned_stats(X_all, Y_all, bins=args.aggregate_bins, out_csv=agg_csv)  # we save binned stats

                # we collect data for consolidated plot (using specific sentence index from first trial)
                total_sentences = args.n_encoder_sentences + args.n_decoder_sentences  # we total sentences
                if args.consolidate_sentence_idx < total_sentences:
                    # we load first trial data for this config
                    first_trial_dots = np.load(data_dir / f"{args.name}_{config_id}_trial0_dots.npy")  # we load dots
                    first_trial_K = np.load(data_dir / f"{args.name}_{config_id}_trial0_K_attn.npy")  # we load K
                    
                    # we extract pairs involving the specific sentence
                    N = first_trial_K.shape[0]  # we size
                    sent_idx = args.consolidate_sentence_idx  # we target sentence
                    other_indices = [i for i in range(N) if i != sent_idx]  # we other sentences
                    
                    x_pairs = []  # we x for this sentence
                    y_pairs = []  # we y for this sentence
                    for j in other_indices:
                        x_pairs.append(first_trial_dots[sent_idx, j])  # we dot with sentence j
                        y_pairs.append(first_trial_K[sent_idx, j])  # we kernel with sentence j
                    
                    if len(x_pairs) > 0:
                        consolidate_data.append({
                            "x_data": np.array(x_pairs, dtype=np.float32),
                            "y_data": np.array(y_pairs, dtype=np.float32),
                            "label": f"h{n_heads}_d{d_model}",
                            "config_id": config_id,
                            "n_heads": n_heads,
                            "d_model": d_model,
                        })  # we add to consolidation

                # we save per-config json with token count info
                config_specific = {
                    "config_id": config_id,
                    "n_heads": n_heads,
                    "d_model": d_model,
                    "d_embed": int(d_embed),
                    "trials": args.trials,
                    "n_encoder_sentences": args.n_encoder_sentences,
                    "n_decoder_sentences": args.n_decoder_sentences,
                    "aggregate_pairs": int(len(X_all)),
                    "architecture": "encoder_decoder_attention",
                    "n_encoder_tokens": args.n_encoder_tokens,
                    "n_decoder_tokens": args.n_decoder_tokens,
                    "min_bucket_size": args.min_bucket_size,
                    "coverage_threshold": args.coverage_threshold,
                    "dot_product_strategy": "controlled_coverage",
                }  # we config dict
                config_json = data_dir / f"{args.name}_{config_id}_config.json"  # we path
                save_config_json(config_specific, config_json)  # we save config

            # we free gpu memory per config
            del attn, fcnn  # we delete models
            torch.cuda.empty_cache()  # we free cache

    # we create consolidated plot for all configs
    if len(consolidate_data) > 0:
        consolidate_plot = plots_dir / f"{args.name}_all_configs_sentence{args.consolidate_sentence_idx}.png"  # we path
        plot_all_configs_consolidated(
            configs_data=consolidate_data,
            out_path=consolidate_plot,
            title=f"NTK vs dot for sentence {args.consolidate_sentence_idx} (all configs)",
            bins=args.aggregate_bins,
        )  # we plot consolidated
        
        # we save consolidation data as json
        consolidate_json = data_dir / f"{args.name}_consolidate_sentence{args.consolidate_sentence_idx}.json"  # we path
        consolidate_export = {
            "sentence_index": args.consolidate_sentence_idx,
            "configs": [
                {
                    "config_id": c["config_id"],
                    "n_heads": c["n_heads"],
                    "d_model": c["d_model"],
                    "n_pairs": len(c["x_data"]),
                } for c in consolidate_data
            ],
        }  # we export dict
        save_config_json(consolidate_export, consolidate_json)  # we save consolidation config

    print("done")  # we signal end


if __name__ == "__main__":
    # we hardcode configuration for optimal dot product coverage
    n_encoder = 10  # we encoder sentences
    n_decoder = 10  # we decoder sentences
    total = n_encoder + n_decoder  # we total sentences
    n_pairs = (total * (total - 1)) // 2  # we pairs calculation
    min_bucket_size = 0.1  # we bucket size
    n_buckets = int(2.0 / min_bucket_size)  # we total buckets
    coverage_pct = (n_pairs / n_buckets) * 100  # we coverage percentage
    points_per_bucket = n_pairs / n_buckets  # we points per bucket
    
    print(f"=== HARDCODED CONFIGURATION ===")  # we header
    print(f"Avec {n_encoder} phrases encoder + {n_decoder} phrases decoder = {total} phrases total")  # we sentences
    print(f"Nombre de paires = C({total},2) = {n_pairs} points")  # we pairs
    print(f"Pour buckets de {min_bucket_size} sur [-1,1] = {n_buckets} buckets")  # we buckets
    print(f"Couverture = {n_pairs}/{n_buckets} = {coverage_pct:.0f}%")  # we coverage
    print(f"Points par bucket en moyenne = {points_per_bucket:.1f}")  # we density
    print(f"===============================")  # we footer
    
    main()  # we run main