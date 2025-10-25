import argparse, json
from pathlib import Path
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.callbacks import CallbackManager

from src.config import settings
from src.models.llm import make_chat_model
from src.chains.fact_check import build_fact_check_chain
from src.data_loaders.politifact import load_politifact
from src.utils.jsonl_writer import JSONLWriter
from src.utils.metrics_schema import FactCheckRecord
from src.utils.timing import timing_ms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=None, help="Override OPENAI_MODEL")
    p.add_argument("--max_records", type=int, default=100, help="limit for quick runs")
    p.add_argument(
        "--split", default="train", choices=["train", "valid", "test", "all"]
    )
    p.add_argument(
        "--prompt", default=str(Path(__file__).parent / "prompts" / "fact_check.txt")
    )
    p.add_argument("--results", default=None, help="Override results file path")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.model or settings.openai_model
    provider = args.provider

    llm = make_chat_model(
        provider=provider, model_name=model_name, api_key=settings.openai_api_key
    )
    chain = build_fact_check_chain(llm, Path(args.prompt))

    results_path = (
        Path(args.results)
        if args.results
        else settings.results_dir / f"{settings.run_name}.jsonl"
    )
    writer = JSONLWriter(results_path)

    data = load_politifact(
        settings.data_dir, split="train" if args.split == "all" else args.split
    )
    if args.max_records:
        data = data[: args.max_records]

    cb = CallbackManager([ConsoleCallbackHandler()])

    for cid, claim, gold in tqdm(data, desc="Fact-checking"):
        with timing_ms() as ms:
            try:
                out = chain.invoke({"claim": claim}, config={"callbacks": cb})
                result = out["result"]  # parsed JSON dict
                # model may sometimes produce strings for confidence
                confidence = float(result.get("confidence", 0))
                verdict = str(result.get("verdict", "")).upper()
                rationale = result.get("rationale", "").strip()
                cited_knowledge = result.get("cited_knowledge", "").strip()
                safety_notes = result.get("safety_notes", "").strip()
            except Exception as e:
                verdict, confidence = "MIXED", 0.0
                rationale = f"Parser/Model error: {e}"
                cited_knowledge, safety_notes = (
                    "",
                    "Model output could not be parsed; defaulting to MIXED.",
                )

        # If token usage is exposed by the client, pull it from LLM metadata; OpenAI SDK v1 puts it on response,
        # but LangChain abstracts it; we keep placeholders for now.
        record = FactCheckRecord(
            run_name=settings.run_name,
            model_name=model_name,
            provider=provider,
            claim_id=cid,
            claim_text=claim,
            gold_label=gold,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            cited_knowledge=cited_knowledge,
            safety_notes=safety_notes,
            latency_ms=ms(),
            extra={},
        )
        writer.write(record.model_dump())

    print(f"Wrote results to: {results_path}")


if __name__ == "__main__":
    main()
