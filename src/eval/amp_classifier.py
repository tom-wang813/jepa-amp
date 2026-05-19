"""
AMP binary classifiers.

Available classifiers:
  - AMPlifyClassifier
      Pre-trained transformer AMP predictor (pip install AMPlify).
  - ESMAMPClassifier
      HuggingFace ESM-2 fine-tuned for AMP sequence classification.
  - JEPAAMPClassifier
      JEPA context encoder features + logistic-regression probe.
      Used to validate JEPA representation quality, not as a standalone classifier.
  - MacrelClassifier
      Wraps the Macrel CLI (pip install macrel).

All classifiers expose:
    clf.predict_proba(sequences: list[str]) -> np.ndarray  # P(AMP), shape (N,)
"""

import csv
import gzip
import logging
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import POSITIVE_AA, NEGATIVE_AA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UniProt negative-sample downloader
# ---------------------------------------------------------------------------

_UNIPROT_REST = "https://rest.uniprot.org/uniprotkb/search"


def _fetch_non_amp_sequences(max_seqs: int = 2000, max_len: int = 50, timeout: int = 60) -> list[str]:
    import re as _re
    import time as _time

    query = f"reviewed:true AND length:[5 TO {max_len}] NOT keyword:KW-0929"
    page_size = 500
    sequences: list[str] = []
    cursor = None
    logger.info("Fetching non-AMP sequences from UniProt (target=%d)…", max_seqs)

    while len(sequences) < max_seqs:
        base = (
            _UNIPROT_REST
            + "?query=" + urllib.parse.quote(query)
            + "&format=fasta"
            + f"&size={page_size}"
        )
        url = base if cursor is None else base + f"&cursor={cursor}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "jepa-amp-eval/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                link_header = resp.headers.get("Link", "")
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            logger.warning("UniProt fetch failed: %s. Stopping pagination.", exc)
            break

        current: list[str] = []
        for line in raw.splitlines():
            if line.startswith(">"):
                if current:
                    seq = "".join(current).upper()
                    if 5 <= len(seq) <= max_len and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                        sequences.append(seq)
                current = []
            else:
                current.append(line.strip())
        if current:
            seq = "".join(current).upper()
            if 5 <= len(seq) <= max_len and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                sequences.append(seq)

        cursor = None
        if 'rel="next"' in link_header:
            m = _re.search(r'cursor=([^&>]+)', link_header)
            if m:
                cursor = m.group(1)
        if cursor is None or raw.count(">") == 0:
            break
        _time.sleep(0.2)

    logger.info("Downloaded %d non-AMP sequences from UniProt.", len(sequences))
    return sequences[:max_seqs]


# ---------------------------------------------------------------------------
# AMPlify classifier  (pre-trained transformer)
# ---------------------------------------------------------------------------

class AMPlifyClassifier:
    """
    Wraps AMPlify (bcgsc/AMPlify) — a transformer-based AMP predictor.
    Install: pip install AMPlify

    Reference: Li et al., Communications Biology 2022.
    """

    def __init__(self):
        self._available = False
        try:
            from AMPlify import AMP_predict  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("AMPlify not installed. Install with: pip install AMPlify")

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        if not self._available:
            return np.full(len(sequences), np.nan, dtype=np.float32)
        if not sequences:
            return np.array([], dtype=np.float32)
        from AMPlify import AMP_predict
        result = AMP_predict(sequences)
        # result is a DataFrame with columns: Sequence, Score, Prediction
        return result["Score"].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# ESM-2 fine-tuned AMP classifier  (HuggingFace)
# ---------------------------------------------------------------------------

class ESMAMPClassifier:
    """
    HuggingFace ESM-2 model fine-tuned for AMP binary classification.

    The model must output 2 logits (class 0 = non-AMP, class 1 = AMP).
    Install: pip install transformers

    Parameters
    ----------
    model_id : str
        HuggingFace model ID of the fine-tuned ESM-2 AMP classifier.
    device : str
        PyTorch device string (e.g. 'cpu', 'cuda:0').
    """

    def __init__(
        self,
        model_id: str = "nuphar/esm2_t6_8M_UR50D_finetuned_AMP",
        device: str = "cpu",
    ):
        self._model_id = model_id
        self._device = torch.device(device)
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError("Install transformers with: pip install transformers")
        logger.info("Loading ESM-2 AMP classifier: %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self._model_id)
            .to(self._device)
            .eval()
        )
        self._loaded = True

    def predict_proba(self, sequences: list[str], batch_size: int = 64) -> np.ndarray:
        if not sequences:
            return np.array([], dtype=np.float32)
        try:
            self._load()
        except Exception as exc:
            logger.warning("ESMAMPClassifier unavailable: %s. Returning NaN.", exc)
            return np.full(len(sequences), np.nan, dtype=np.float32)

        all_probs: list[float] = []
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            # ESM tokenizers expect space-separated amino acids
            spaced = [" ".join(seq) for seq in chunk]
            inputs = self._tokenizer(
                spaced,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits  # (B, 2)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().float().tolist()
            all_probs.extend(probs)

        return np.array(all_probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# JEPA-encoder based classifier  (evaluates representation quality)
# ---------------------------------------------------------------------------

class JEPAAMPClassifier:
    """
    Uses the pre-trained JEPA context encoder as a fixed feature extractor
    (mean-pool over residue positions), then fits a logistic regression.

    This measures whether JEPA learned useful AMP representations vs a random
    encoder — not intended as a production AMP classifier.
    """

    def __init__(self, encoder, device: str = "cpu"):
        self._encoder = encoder
        self._device = torch.device(device)
        self._encoder.to(self._device).eval()
        self._pipeline: Pipeline | None = None
        self._fitted = False
        self._pos_class_idx = 1

    def _embed(self, sequences: list[str], batch_size: int = 256) -> np.ndarray:
        from src.data.tokenizer import encode, PAD_ID

        all_embs = []
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            max_len = max(len(encode(s, add_special_tokens=True)) for s in chunk)
            tokens = torch.full((len(chunk), max_len), PAD_ID, dtype=torch.long)
            lengths = []
            for j, seq in enumerate(chunk):
                ids = encode(seq, add_special_tokens=True)
                tokens[j, : len(ids)] = torch.tensor(ids)
                lengths.append(len(ids))
            tokens = tokens.to(self._device)
            with torch.no_grad():
                h = self._encoder(tokens)  # (B, L, D)
            for j, L in enumerate(lengths):
                emb = h[j, 1 : L - 1].mean(0).cpu().float().numpy()
                all_embs.append(emb)
        return np.stack(all_embs)

    def fit(self, pos_seqs: list[str], neg_seqs: list[str]) -> "JEPAAMPClassifier":
        if not neg_seqs:
            neg_seqs = _fetch_non_amp_sequences(max_seqs=len(pos_seqs) * 2)
        if not neg_seqs:
            import random
            neg_seqs = ["".join(random.sample(list(s), len(s))) for s in pos_seqs]

        logger.info("JEPAAMPClassifier: embedding %d sequences …", len(pos_seqs) + len(neg_seqs))
        X = np.vstack([self._embed(pos_seqs), self._embed(neg_seqs)])
        y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs), dtype=np.int32)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ])
        self._pipeline.fit(X, y)
        classes = list(self._pipeline.named_steps["clf"].classes_)
        self._pos_class_idx = classes.index(1) if 1 in classes else 1
        self._fitted = True
        logger.info("JEPAAMPClassifier fitted.")
        return self

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        proba = self._pipeline.predict_proba(self._embed(sequences))
        return proba[:, self._pos_class_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Macrel classifier (external CLI wrapper)
# ---------------------------------------------------------------------------

class MacrelClassifier:
    """
    Wraps the Macrel CLI tool for AMP prediction.
    Install: pip install macrel

    Reference: Santos-Júnior et al., Genome Biology 2020.
    """

    def __init__(self, threads: int = 4):
        self._threads = threads

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(["macrel", "--version"], capture_output=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        if not self._check_available():
            logger.warning("Macrel not found. Install with: pip install macrel")
            return np.full(len(sequences), np.nan, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fasta_path = tmpdir / "input.fasta"

            with open(fasta_path, "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">seq_{i:06d}\n{seq}\n")

            result = subprocess.run(
                ["macrel", "seq", "-f", str(fasta_path), "-o", str(tmpdir),
                 "--threads", str(self._threads)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.error("Macrel failed (returncode=%d):\n%s", result.returncode, result.stderr)
                return np.full(len(sequences), np.nan, dtype=np.float32)

            out_file = tmpdir / "macrel.prediction.gz"
            if not out_file.exists():
                logger.error("Macrel output file not found: %s", out_file)
                return np.full(len(sequences), np.nan, dtype=np.float32)

            scores_dict: dict[str, float] = {}
            with gzip.open(out_file, "rt") as f:
                reader = csv.reader(f, delimiter="\t")
                header_seen = False
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    if not header_seen:
                        header_seen = True
                        continue
                    seq_id = row[0]
                    try:
                        scores_dict[seq_id] = float(row[2])
                    except (IndexError, ValueError):
                        pass

            scores = np.array(
                [scores_dict.get(f"seq_{i:06d}", np.nan) for i in range(len(sequences))],
                dtype=np.float32,
            )
            n_nan = np.isnan(scores).sum()
            if n_nan > 0:
                logger.warning("Macrel returned NaN for %d / %d sequences.", n_nan, len(sequences))
            logger.info("Macrel scored %d sequences.", len(sequences) - n_nan)
            return scores
