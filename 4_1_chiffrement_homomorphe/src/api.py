from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from concrete.ml.deployment import FHEModelServer

app = FastAPI(title="FHE Credit Risk Server (Blind Backend)")

_root = Path(__file__).resolve().parents[1]
_artifacts = _root / "artifacts"
_fhe_dir = _artifacts / "fhe"

server = FHEModelServer(path_dir=str(_fhe_dir))
server.load()


class FHERunRequest(BaseModel):
    encrypted_data_b64: str
    evaluation_keys_b64: str


class FHERunResponse(BaseModel):
    encrypted_result_b64: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run_fhe", response_model=FHERunResponse)
def run_fhe(req: FHERunRequest):
    print(f"encrypted data received from the the frontend in the payload : {req}")
    encrypted_data = base64.b64decode(req.encrypted_data_b64)
    eval_keys = base64.b64decode(req.evaluation_keys_b64)

    encrypted_result = server.run(
        serialized_encrypted_quantized_data=encrypted_data,
        serialized_evaluation_keys=eval_keys,
    )

    return FHERunResponse(encrypted_result_b64=base64.b64encode(encrypted_result).decode("ascii"))
