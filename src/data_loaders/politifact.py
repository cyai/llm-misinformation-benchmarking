from pathlib import Path
import pandas as pd
import json


def load_politifact(data_dir: Path, split: str = "train"):
    """
    Expects the dataset downloaded into:
      {data_dir}/politifact
    with JSON file named politifact_factcheck_data.json.
    The dataset has fields like: 'verdict', 'statement', 'statement_originator', etc.
    """
    base = Path(data_dir) / "politifact"
    json_file = base / "politifact_factcheck_data.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"Could not find Politifact JSON file at {json_file}. Place the file there."
        )

    # Load JSON data (assuming JSONL format - one JSON object per line)
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    df = pd.DataFrame(data)

    # Normalize likely column names - map JSON fields to expected columns
    # Based on the JSON structure: verdict, statement, statement_originator, etc.
    id_col = None  # Generate IDs since JSON doesn't have explicit ID field
    text_col = "statement"  # The claim text
    label_col = "verdict"  # The fact-check verdict

    # Verify required columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Missing required columns in JSON. Expected 'statement' and 'verdict'. Found: {df.columns.tolist()}"
        )

    # Some releases have many label granularities (pants-fire, false, half-true, etc.)
    # We map them to {FACT, MIXED, FALSE} for this run; adjust mapping as you like.
    def map_label(x):
        if label_col is None:
            return None
        xl = str(x).strip().lower()
        if xl in {"true", "mostly-true"}:
            return "FACT"
        if xl in {"half-true", "barely-true"}:
            return "MIXED"
        if xl in {"false", "pants-fire", "mostly-false"}:
            return "FALSE"
        # unseen/other
        return None

    out = []
    for idx, row in df.iterrows():
        # Generate ID since JSON doesn't have explicit ID field
        cid = str(idx)
        claim = str(row[text_col])
        gold = map_label(row[label_col])
        out.append((cid, claim, gold))
    return out
