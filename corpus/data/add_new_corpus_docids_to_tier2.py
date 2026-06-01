import argparse
import json
import os
from typing import Iterable, List, Set


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_mapping(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Existing mapping not found: {path}")
    mapping = _read_json(path)
    if not isinstance(mapping, dict):
        raise ValueError(f"Existing mapping must be a JSON object (dict). Got: {type(mapping)}")
    return mapping


def iter_docids_from_txt(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield s


def iter_docids_from_json(path: str, field: str) -> Iterable[str]:
    obj = _read_json(path)
    if isinstance(obj, list):
        for x in obj:
            if x is None:
                continue
            yield str(x)
        return

    if isinstance(obj, dict):
        if field in obj and isinstance(obj[field], list):
            for x in obj[field]:
                if x is None:
                    continue
                yield str(x)
            return

    raise ValueError(
        "JSON docids file must be either a list of docids, "
        "or a dict containing a list under --field."
    )


def iter_docids_from_jsonl(path: str, field: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                continue
            if field in obj:
                yield str(obj[field])
            elif field == "docid" and "id" in obj:
                yield str(obj["id"])


def load_new_docids(path: str, fmt: str, field: str) -> List[str]:
    fmt = fmt.lower()
    if fmt == "auto":
        if path.endswith(".jsonl"):
            fmt = "jsonl"
        elif path.endswith(".json"):
            fmt = "json"
        else:
            fmt = "txt"

    if fmt == "txt":
        return list(iter_docids_from_txt(path))
    if fmt == "json":
        return list(iter_docids_from_json(path, field))
    if fmt == "jsonl":
        return list(iter_docids_from_jsonl(path, field))

    raise ValueError(f"Unsupported format: {fmt}. Use one of: auto, txt, json, jsonl")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add docids from a new corpus into an existing docid->tier mapping, "
            "assigning all added docids to a fixed tier (default: tier 2)."
        )
    )
    parser.add_argument(
        "--existing_mapping",
        type=str,
        required=True,
        help="Path to existing docid_tier_mapping.json",
    )
    parser.add_argument(
        "--new_docids",
        type=str,
        required=True,
        help="Path to docids list file (txt/json/jsonl)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "txt", "json", "jsonl"],
        help="Format of --new_docids (default: auto by extension)",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="docid",
        help="Field name for json/jsonl inputs (default: docid). For json, expects dict[field]=list.",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=2,
        help="Tier value to assign for all newly added docids (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for updated mapping. Default: overwrite --existing_mapping.",
    )
    parser.add_argument(
        "--override_existing",
        action="store_true",
        help="If set, overwrite tier for docids that already exist in mapping.",
    )

    args = parser.parse_args()

    existing = load_existing_mapping(args.existing_mapping)
    new_docids = load_new_docids(args.new_docids, args.format, args.field)

    new_set: Set[str] = {str(d) for d in new_docids if str(d).strip()}
    before_n = len(existing)

    updated = dict(existing)
    added = 0
    overwritten = 0

    for d in new_set:
        if d in updated:
            if args.override_existing and int(updated[d]) != int(args.tier):
                updated[d] = int(args.tier)
                overwritten += 1
        else:
            updated[d] = int(args.tier)
            added += 1

    out_path = args.output or args.existing_mapping
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    after_n = len(updated)
    print("=" * 60)
    print("Update docid->tier mapping")
    print("=" * 60)
    print(f"Existing mapping: {args.existing_mapping}")
    print(f"New docids file : {args.new_docids} (format={args.format}, field={args.field})")
    print(f"Tier for new     : {args.tier}")
    print(f"Before size      : {before_n:,}")
    print(f"Unique new docids: {len(new_set):,}")
    print(f"Added docids     : {added:,}")
    print(f"Overwritten      : {overwritten:,} (override_existing={args.override_existing})")
    print(f"After size       : {after_n:,}")
    print(f"Saved to         : {out_path}")


if __name__ == "__main__":
    main()
