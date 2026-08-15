"""Generates the Python gRPC stubs from proto/search.proto into
py/server/generated/ (gitignored — regenerate with `python -m server.gen_proto`
whenever proto/search.proto changes, the same relationship the C++ side has
to its own generated code under cpp/build/generated/).
"""

from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = Path(__file__).resolve().parent / "generated"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()
    args = [
        "protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        str(PROTO_DIR / "search.proto"),
    ]
    code = protoc.main(args)
    if code != 0:
        raise SystemExit(f"protoc failed with code {code}")
    print(f"wrote {OUT_DIR / 'search_pb2.py'} and {OUT_DIR / 'search_pb2_grpc.py'}")


if __name__ == "__main__":
    main()
