from pathlib import Path
import json


class JSONLWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

    def write(self, obj) -> None:
        if self.file:
            self.file.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.file.flush()
        else:
            # Fallback for non-context-manager usage
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
