import csv
from pathlib import Path


INPUT_CSV = Path("ari_llm_outputs.csv")


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {csv_path}")

        rows = [{key: (value or "") for key, value in row.items()} for row in reader]

    return rows


def print_results(rows: list[dict[str, str]]) -> None:
    for row in rows:
        print(f"file: {row.get('prompt_file', '')}")
        print(f"output: {row.get('output', '')}")
        print()


def main() -> None:
    rows = read_rows(INPUT_CSV)
    print_results(rows)


if __name__ == "__main__":
    main()
