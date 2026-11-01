"""One-time explicit download. Query/index commands subsequently run offline."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[1]
path = root / "config/settings.json"
config = json.loads(path.read_text())
revision = config["model_revision"] or HfApi().model_info(config["model_id"]).sha
target = root / "data/models/multilingual-e5-small"
snapshot_download(repo_id=config["model_id"], revision=revision, local_dir=target,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "1_Pooling/*"],
                  ignore_patterns=["onnx/*", "openvino/*"])
(target / "finrag-model.json").write_text(json.dumps({"model_id": config["model_id"], "revision": revision}, indent=2))
config["model_revision"] = revision
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"Model downloaded and pinned: {config['model_id']} @ {revision}")
