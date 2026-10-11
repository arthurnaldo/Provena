"""SSCD frame embeddings (Meta, MIT licence). Needs PyTorch, which is not a package dependency:
run with `uv run --with torch ...`. Never import this in a process that also loads FAISS (both bundle
OpenMP and crash together).
"""

import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
COPY_THRESHOLD = 0.75  # cosine similarity; Meta's DISC operating point (~90% precision per frame)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class SSCD:
    def __init__(self, model_path: str | Path, device: str | None = None):
        import torch
        model_path = Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, model_path)
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model_path)).eval().to(self.device)

    def embed(self, frames: np.ndarray, batch: int = 128) -> np.ndarray:
        """(n, 320, 320, 3) uint8 RGB → (n, 512) float16, L2-normalised."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                x = (frames[i:i + batch].astype(np.float32) / 255 - MEAN) / STD
                x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
                out.append(torch.nn.functional.normalize(self.model(x), dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float16) if out else np.zeros((0, 512), np.float16)
