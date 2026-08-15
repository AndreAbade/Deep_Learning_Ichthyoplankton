import argparse
import csv
import os
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "DataSet"
MAPPING_CSV = DATASET_DIR / "rename_mapping.csv"


def collect_renames(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    renames = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() == ".jpg")
        for i, old_path in enumerate(files, start=1):
            new_name = f"img_{i:05d}.jpg"
            new_path = class_dir / new_name
            renames.append((class_dir.name, old_path, new_path))
    return renames


def dry_run(renames: list[tuple[str, Path, Path]]) -> None:
    current_class = None
    for classe, old, new in renames:
        if classe != current_class:
            current_class = classe
            print(f"\n[{classe}]  ({sum(1 for c, _, _ in renames if c == classe)} arquivos)")
            print(f"  Primeiro: {old.name}  ->  {new.name}")
        last = (classe, old, new)
    # print last of each class
    seen = set()
    for classe, old, new in reversed(renames):
        if classe not in seen:
            seen.add(classe)
            print(f"  Último:   {old.name}  ->  {new.name}")
    print(f"\nTotal: {len(renames)} arquivos seriam renomeados.")


def execute(renames: list[tuple[str, Path, Path]]) -> None:
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["classe", "nome_original", "nome_novo"])
        for classe, old, new in renames:
            writer.writerow([classe, old.name, new.name])

    renamed = 0
    for _, old, new in renames:
        old.rename(new)
        renamed += 1

    print(f"Renomeados: {renamed} arquivos.")
    print(f"Mapeamento salvo em: {MAPPING_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Remove nomes de classe dos arquivos do DataSet.")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra o que seria feito.")
    args = parser.parse_args()

    renames = collect_renames(DATASET_DIR)

    if args.dry_run:
        dry_run(renames)
    else:
        execute(renames)


if __name__ == "__main__":
    main()
