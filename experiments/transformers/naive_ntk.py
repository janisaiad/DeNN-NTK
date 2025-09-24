import argparse
import os
from pathlib import Path
import random
from typing import List, Tuple, Dict

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


def filter_vocab_words(tokenizer: AutoTokenizer, max_words: int = 20000) -> List[str]:
    """we pick word-level tokens (no '##', alphabetic) to build 10-word sentences"""
    vocab_items = list(tokenizer.get_vocab().items())  # we list vocab
    vocab_items.sort(key=lambda kv: kv[1])  # we sort by id
    candidates: List[str] = []  # we prepare list
    for tok, _ in vocab_items[:max_words]:
        if tok.startswith("##"):  # we skip subword continuations
            continue  # we continue
        t = tok.strip()  # we clean
        if t in tokenizer.all_special_tokens:  # we skip special tokens
            continue  # we continue
        if any(ch.isdigit() for ch in t):  # we skip digits
            continue  # we continue
        if not any(ch.isalpha() for ch in t):  # we require alphabetic
            continue  # we continue
        candidates.append(t)  # we add token
    if len(candidates) < 1000:
        candidates = [tok for tok, _ in vocab_items if tok not in tokenizer.all_special_tokens][:5000]  # we fallback
    return candidates  # we return candidates


def generate_sentences(tokenizer: AutoTokenizer, n_sentences: int, n_words: int, seed: int) -> List[str]:
    """we generate n_sentences random sentences of exactly n_words words from vocab"""
    rng = random.Random(seed)  # we create rng
    words = filter_vocab_words(tokenizer)  # we filter vocab
    sents: List[str] = []  # we init list
    for _ in range(n_sentences):
        chosen = rng.sample(words, k=n_words)  # we sample words
        sent = " ".join(chosen)  # we build sentence
        sents.append(sent)  # we append
    return sents  # we return sentences


def sentence_token_ids(tokenizer: AutoTokenizer, sentence: str) -> List[int]:
    """we tokenize a sentence as split by space and keep first subtoken id for each word to stay at length 10"""
    pieces = sentence.split()  # we split by spaces
    ids: List[int] = []  # we init list
    for w in pieces:
        toks = tokenizer.tokenize(w)  # we tokenize word
        if len(toks) == 0:
            toks = [tokenizer.unk_token]  # we fallback unk
        tok_ids = tokenizer.convert_tokens_to_ids(toks)  # we convert to ids
        ids.append(int(tok_ids[0]))  # we keep first subword
    return ids  # we return ids


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


class AttentionOneLayer(nn.Module):
    """we implement a single-layer multi-head attention + readout to scalar"""

    def __init__(self, d_embed: int, d_model: int, n_heads: int):
        super().__init__()  # we init parent
        assert d_model % n_heads == 0, "we require d_model divisible by n_heads"  # we check divisibility
        self.d_embed = d_embed  # we store
        self.d_model = d_model  # we store
        self.n_heads = n_heads  # we store
        self.d_head = d_model // n_heads  # we compute head dim

        self.proj_in = nn.Linear(d_embed, d_model, bias=False)  # we project input
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # we query matrix
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # we key matrix
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # we value matrix
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # we output projection
        self.readout = nn.Linear(d_model, 1, bias=False)  # we scalar readout

        self.reset_parameters()  # we init weights

    def reset_parameters(self) -> None:
        """we use fan-in scaling to keep gradients stable"""
        for m in [self.proj_in, self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear", mode="fan_in")  # we init linear weights
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / np.sqrt(self.d_model))  # we init readout

    def forward(self, X_tokens: torch.Tensor) -> torch.Tensor:
        """we forward a [T, d_embed] token matrix and return a scalar"""
        T = X_tokens.shape[0]  # we get length
        H = self.n_heads  # we number of heads
        Dh = self.d_head  # we head dim

        H_in = self.proj_in(X_tokens)  # we project to model dim [T, d_model]
        Q = self.W_q(H_in)  # we compute queries [T, d_model]
        K = self.W_k(H_in)  # we compute keys [T, d_model]
        V = self.W_v(H_in)  # we compute values [T, d_model]

        def split_heads(Z: torch.Tensor) -> torch.Tensor:
            Z = Z.view(T, H, Dh).transpose(0, 1).contiguous()  # we get [H, T, Dh]
            return Z  # we return split

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)  # we split heads

        attn_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) / np.sqrt(Dh)  # we compute logits [H, T, T]
        attn_weights = F.softmax(attn_logits, dim=-1)  # we softmax over keys [H, T, T]
        context = torch.matmul(attn_weights, Vh)  # we apply attention [H, T, Dh]

        context = context.transpose(0, 1).contiguous().view(T, H * Dh)  # we merge heads [T, d_model]
        out = self.W_o(context)  # we project out [T, d_model]
        h = out.mean(dim=0, keepdim=False)  # we mean pool [d_model]
        y = self.readout(h)  # we read scalar [1]
        return y.squeeze(0)  # we return scalar


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
    # we open csv for appending with header
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("label,i,j,K_ij\n")  # we write header
    # we compute gradients and fill K progressively
    for i in tqdm(range(N), desc=f"grads[{label}]", leave=False):
        g_i = grad_wrt_params(model, token_seqs[i].to(device))  # we compute gradient for i
        G.append(g_i)  # we store gradient
        gi_np = g_i.numpy()  # we to numpy
        # we update diagonal
        K[i, i] = float(np.dot(gi_np, gi_np))  # we set diagonal
        # we update row with previous j
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


def scatter_with_binning(ax, x: np.ndarray, y: np.ndarray, color: str, name: str, bins: int = 20) -> None:
    """we draw scatter and binned means"""
    ax.scatter(x, y, s=6, alpha=0.25, color=color, label=f"{name} (pairs)")  # we scatter raw pairs
    if len(x) == 0:
        return  # we handle empty
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
        ax.plot(means_x, means_y, color=color, linewidth=2.0, label=f"{name} (binned)")  # we plot binned curve


def plot_ntk_vs_dot(
    s_pairs: np.ndarray,
    K_attn: np.ndarray,
    K_fcnn: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """we plot K(x,y) as function of dot(x,y) for attention and fcnn"""
    # we take off-diagonal pairs
    N = K_attn.shape[0]  # we infer size
    iu, ju = np.triu_indices(N, k=1)  # we take upper tri
    x = s_pairs[iu, ju].astype(np.float32)  # we x values
    y_a = K_attn[iu, ju].astype(np.float32)  # we attn kernel
    y_f = K_fcnn[iu, ju].astype(np.float32)  # we fcnn kernel

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0), dpi=140)  # we create figure
    scatter_with_binning(ax, x, y_a, color="#1f77b4", name="attention")  # we plot attention
    scatter_with_binning(ax, x, y_f, color="#ff7f0e", name="fcnn")  # we plot fcnn
    ax.set_xlabel("dot(x, y) on unit sphere")  # we label x
    ax.set_ylabel("NTK(x, y)")  # we label y
    ax.set_title(title)  # we set title
    ax.grid(True, alpha=0.25)  # we add grid
    ax.legend()  # we add legend
    fig.tight_layout()  # we tighten
    fig.savefig(out_path)  # we save figure
    plt.close(fig)  # we close figure


def main():
    parser = argparse.ArgumentParser(description="we compute empirical ntk for 1-layer attention vs fcnn")  # we parser
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="we choose hf model name")  # we arg
    parser.add_argument("--n_sentences", type=int, default=40, help="we number of sentences to sample")  # we arg
    parser.add_argument("--n_words", type=int, default=10, help="we words per sentence")  # we arg
    parser.add_argument("--heads", type=str, default="1,2,4,8", help="we comma list of head counts")  # we arg
    parser.add_argument("--dims", type=str, default="64,128,256", help="we comma list of internal d_model")  # we arg
    parser.add_argument("--seed", type=int, default=123, help="we random seed")  # we arg
    parser.add_argument("--device", type=str, default="cuda", help="we device cuda or cpu")  # we arg
    parser.add_argument("--name", type=str, default="orion_ntk", help="we run name used in filenames")  # we arg
    args = parser.parse_args()  # we parse

    set_seed(args.seed)  # we set seed

    dev = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")  # we pick device

    this_file = Path(__file__)  # we locate this file
    plots_dir, data_dir = resolve_paths(this_file)  # we resolve output dirs

    tokenizer, emb_matrix = load_hf_embeddings(args.model_name, dev)  # we load tokenizer and embeddings

    # we generate sentences
    sentences = generate_sentences(tokenizer, args.n_sentences, args.n_words, seed=args.seed + 7)  # we gen sents

    # we embed and normalize tokens for each sentence
    token_seqs: List[torch.Tensor] = []  # we store token embeddings sequences
    sent_vectors: List[torch.Tensor] = []  # we store mean unit vectors
    for s in tqdm(sentences, desc="embed+normalize", leave=False):
        ids = sentence_token_ids(tokenizer, s)  # we get word token ids
        X = embed_and_normalize_tokens(ids, emb_matrix)  # we get normalized embeddings [T,d_embed]
        token_seqs.append(X)  # we store tokens
        sent_vectors.append(sentence_mean_unit_vector(X))  # we store mean vector

    # we compute sentence dot products matrix
    with torch.no_grad():
        S = torch.stack(sent_vectors, dim=0)  # we stack [N,d]
        S = F.normalize(S, p=2.0, dim=-1)  # we ensure unit
        dots = (S @ S.t()).detach().cpu().numpy().astype(np.float32)  # we compute pairwise dot products

    # we save sentence list and dot products early
    sent_txt = data_dir / f"{args.name}_sentences.txt"  # we path text
    with open(sent_txt, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")  # we save sentences
    dots_path = data_dir / f"{args.name}_dots.npy"  # we path dots
    np.save(dots_path, dots)  # we save dots

    # we iterate over configurations
    heads_list = [int(x.strip()) for x in args.heads.split(",") if x.strip()]  # we parse heads
    dims_list = [int(x.strip()) for x in args.dims.split(",") if x.strip()]  # we parse dims

    d_embed = emb_matrix.shape[1]  # we get embedding dimension

    for d_model in dims_list:
        for n_heads in heads_list:
            if d_model % n_heads != 0:
                continue  # we skip invalid combos
            config_id = f"h{n_heads}_d{d_model}"  # we id config
            print(f"[config] {config_id}")  # we log

            # we instantiate models
            attn = AttentionOneLayer(d_embed=d_embed, d_model=d_model, n_heads=n_heads).to(dev)  # we make attn
            fcnn = FCNNBaseline(d_embed=d_embed, d_model=d_model).to(dev)  # we make fcnn
            attn.eval(), fcnn.eval()  # we eval

            # we paths for kernels and csv logs
            K_attn_path = data_dir / f"{args.name}_{config_id}_K_attn.dat"  # we path memmap
            K_fcnn_path = data_dir / f"{args.name}_{config_id}_K_fcnn.dat"  # we path memmap
            csv_attn_path = data_dir / f"{args.name}_{config_id}_pairs_attn.csv"  # we path csv
            csv_fcnn_path = data_dir / f"{args.name}_{config_id}_pairs_fcnn.csv"  # we path csv

            # we compute ntk incrementally for attention
            K_attn = compute_ntk_incremental(attn, token_seqs, dev, K_attn_path, csv_attn_path, label="attn")  # we run

            # we compute ntk incrementally for fcnn
            K_fcnn = compute_ntk_incremental(fcnn, token_seqs, dev, K_fcnn_path, csv_fcnn_path, label="fcnn")  # we run

            # we also save .npy snapshots
            K_attn_npy = data_dir / f"{args.name}_{config_id}_K_attn.npy"  # we path npy
            K_fcnn_npy = data_dir / f"{args.name}_{config_id}_K_fcnn.npy"  # we path npy
            np.save(K_attn_npy, np.array(K_attn, copy=True))  # we save attn npy
            np.save(K_fcnn_npy, np.array(K_fcnn, copy=True))  # we save fcnn npy

            # we plot K vs dot
            plot_path = plots_dir / f"{args.name}_{config_id}_K_vs_dot.png"  # we path figure
            plot_ntk_vs_dot(
                s_pairs=dots,
                K_attn=np.array(K_attn, copy=False),
                K_fcnn=np.array(K_fcnn, copy=False),
                out_path=plot_path,
                title=f"NTK vs dot on unit sphere ({config_id})",
            )  # we plot and save

            # we free gpu memory
            del attn, fcnn  # we delete models
            torch.cuda.empty_cache()  # we free cache

    print("done")  # we signal end


if __name__ == "__main__":
    main()  # we run main