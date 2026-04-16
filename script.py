import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Tuple

# -------------------------
# User-configurable settings
# -------------------------
#PROVIDER = "hf"
#MODEL_ID = "Qwen/Qwen3-0.6B"
#USE_STRUCTURED_OUTPUT = False

PROVIDER = "ollama"
MODEL_ID = "gpt-oss:120b"
USE_STRUCTURED_OUTPUT = True

PROMPTS_DIR = Path("prompts")
OUTPUT_CSV = Path("ari_llm_outputs.csv")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
OLLAMA_URL = "http://localhost:11434"

def load_prompts(prompts_dir: Path) -> List[Tuple[str, str]]:
    prompt_files = sorted(prompts_dir.glob("*.txt"))
    prompts: List[Tuple[str, str]] = []

    for file_path in prompt_files:
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            prompts.append((file_path.name, content))

    return prompts


def generate_with_huggingface(
    prompts: List[Tuple[str, str]],
    model: str,
    max_new_tokens: int,
    temperature: float,
) -> List[Tuple[str, str]]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers torch"
        ) from exc

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else -1
    generator = pipeline("text-generation", model=model, device=device)
    print(f"[HF] Using {'gpu' if use_gpu else 'cpu'}")
    rows: List[Tuple[str, str]] = []

    for prompt_file, prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = result[0]["generated_text"]
        rows.append((prompt_file, generated))

    return rows


def generate_with_ollama(
    prompts: List[Tuple[str, str]],
    model: str,
    max_new_tokens: int,
    temperature: float,
    ollama_url: str,
    use_structured_output: bool,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    effective_temperature = 0.0 if use_structured_output else temperature

    if use_structured_output:
        try:
            from ollama import chat
        except ImportError as exc:
            raise RuntimeError("ollama is not installed. Run: pip install ollama") from exc

        try:
            from pydantic import BaseModel, ValidationError, model_validator
        except ImportError as exc:
            raise RuntimeError("pydantic is not installed. Run: pip install pydantic") from exc

        class PromptOutput(BaseModel):
            bio_aid_id: str
            ari_label: Literal["Yes", "No"]
            viral_aetiology_label: Literal[
                "Definite", "Probable", "Unlikely", "Negative", "NA"
            ]
            bacterial_aetiology_label: Literal[
                "Definite", "Probable", "Unlikely", "Negative", "NA"
            ]

            @model_validator(mode="after")
            def validate_non_ari_labels_for_no(self) -> "PromptOutput":
                if self.ari_label == "No":
                    if self.viral_aetiology_label != "NA":
                        raise ValueError(
                            "viral_aetiology_label must be 'NA' when ari_label is 'No'"
                        )
                    if self.bacterial_aetiology_label != "NA":
                        raise ValueError(
                            "bacterial_aetiology_label must be 'NA' when ari_label is 'No'"
                        )
                return self

        for prompt_file, prompt in prompts:
            response = chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # Keep the same schema in the prompt text and in `format`
                # for better grounding.
                format=PromptOutput.model_json_schema(),
                options={
                    "num_predict": max_new_tokens,
                    "temperature": effective_temperature,
                },
            )
            try:
                structured = PromptOutput.model_validate_json(response.message.content)
            except ValidationError:
                raw_response_dump = str(
                    response.model_dump() if hasattr(response, "model_dump") else response
                )
                # Attempt a minimal repair for the known business rule:
                # if ari_label is "No", both aetiology labels must be "NA".
                raw_content = (response.message.content or "").strip()
                if not raw_content:
                    print(
                        f"[OLLAMA][STRUCTURED][WARN] Empty response for {prompt_file}. "
                        f"raw_response={raw_response_dump}"
                    )
                    rows.append(
                        (
                            prompt_file,
                            json.dumps(
                                {
                                    "error": "empty_structured_response",
                                    "raw_response": response.message.content,
                                    "response_debug": raw_response_dump,
                                }
                            ),
                        )
                    )
                    continue

                try:
                    repaired = json.loads(raw_content)
                except json.JSONDecodeError:
                    print(
                        f"[OLLAMA][STRUCTURED][WARN] Invalid JSON response for {prompt_file}. "
                        f"raw_response={raw_response_dump}"
                    )
                    rows.append(
                        (
                            prompt_file,
                            json.dumps(
                                {
                                    "error": "invalid_json_structured_response",
                                    "raw_response": response.message.content,
                                    "response_debug": raw_response_dump,
                                }
                            ),
                        )
                    )
                    continue

                if repaired.get("ari_label") == "No":
                    repaired["viral_aetiology_label"] = "NA"
                    repaired["bacterial_aetiology_label"] = "NA"

                try:
                    structured = PromptOutput.model_validate(repaired)
                except ValidationError as exc:
                    print(
                        f"[OLLAMA][STRUCTURED][WARN] Schema validation failed for {prompt_file}. "
                        f"error={exc}"
                    )
                    rows.append(
                        (
                            prompt_file,
                            json.dumps(
                                {
                                    "error": "structured_validation_failed",
                                    "validation_error": str(exc),
                                    "raw_response": response.message.content,
                                    "response_debug": raw_response_dump,
                                }
                            ),
                        )
                    )
                    continue
            rows.append((prompt_file, structured.model_dump_json()))
    else:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed. Run: pip install requests") from exc

        endpoint = f"{ollama_url.rstrip('/')}/api/generate"

        for prompt_file, prompt in prompts:
            response = requests.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            rows.append((prompt_file, data.get("response", "")))

    return rows


def write_csv(
    rows: List[Tuple[str, str]],
    output_csv: Path,
    provider: str,
    model_id: str,
    use_structured_output: bool,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    should_write_header = (not file_exists) or output_csv.stat().st_size == 0

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if should_write_header:
            writer.writerow(
                [
                    "prompt_file",
                    "timestamp",
                    "provider",
                    "model_id",
                    "use_structured_output",
                    "output",
                ]
            )

        for prompt_file, output in rows:
            timestamp = datetime.now(timezone.utc).isoformat()
            writer.writerow(
                [
                    prompt_file,
                    timestamp,
                    provider,
                    model_id,
                    use_structured_output,
                    output,
                ]
            )


def main() -> None:
    provider = PROVIDER.strip().lower()
    model_id = MODEL_ID.strip()
    prompts_dir = PROMPTS_DIR
    output_csv = OUTPUT_CSV

    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompts = load_prompts(prompts_dir)
    if not prompts:
        raise ValueError(f"No non-empty .txt prompts found in: {prompts_dir}")

    if provider == "hf":
        rows = generate_with_huggingface(
            prompts=prompts,
            model=model_id,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
    elif provider == "ollama":
        rows = generate_with_ollama(
            prompts=prompts,
            model=model_id,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            ollama_url=OLLAMA_URL,
            use_structured_output=USE_STRUCTURED_OUTPUT,
        )
    else:
        raise ValueError("PROVIDER must be either 'hf' or 'ollama'")

    write_csv(
        rows=rows,
        output_csv=output_csv,
        provider=provider,
        model_id=model_id,
        use_structured_output=USE_STRUCTURED_OUTPUT,
    )
    print(f"Appended {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()