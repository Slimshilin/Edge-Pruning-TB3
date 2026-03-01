"""
Generate a mixed-entity IOI dataset for the gpt-ioi-circuit task.

Pipeline:
  1. Load the fahamu/ioi dataset and split by held-out templates
     (4 templates reserved for test to create distribution shift).
  2. Replace human names with entities from 6 categories (round-robin):
     names, animals, cities, colors, objects, ambiguous.
     All 4 sampled entities (A, B, C, D) come from the same category.
  3. Build clean sentences (AABB: both occurrences use A and B) and
     corrupted sentences (ABCD: second occurrences replaced with C and D).
     Only keep pairs where GPT-2 token counts match.
  4. Save as HuggingFace DatasetDict with train/validation/test splits.

Preparation log: data/scripts/logs/prepare_ioi_mixed.log
"""

import os
import sys
import json
import random
from tqdm import tqdm
import argparse

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", "-o", default="data/datasets/ioi_mixed")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--start-size", "-n", type=int, default=300000)
    parser.add_argument("--num_test_templates", "-nt", type=int, default=4)
    parser.add_argument("--train", "-t", type=int, default=200)
    parser.add_argument("--validation", "-v", type=int, default=200)
    parser.add_argument("--test", "-e", type=int, default=100)
    parser.add_argument("--tokenizer", "-tk", default="gpt2")
    parser.add_argument("--helper-dir", default="data/helper_files")

    args = parser.parse_args()

    # Load all entity categories
    args.entities = {}
    entity_files = {
        "names": "names.json",
        "animals": "animals.json",
        "cities": "cities.json",
        "colors": "colors.json",
        "objects": "objects.json",
        "ambiguous": "ambiguous.json",
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for category, filename in entity_files.items():
        path = os.path.join(args.helper_dir, filename)
        data = json.load(open(path))
        if "boys" in data:
            raw = data["boys"] + data["girls"]
        else:
            raw = list(data.values())[0]
        # Keep only single-token entities (with leading space, as they appear in sentences)
        valid = [e for e in raw if " " not in e.strip() and len(tokenizer.encode(" " + e)) == 1]
        args.entities[category] = valid

    # Cross-category deduplication: non-name categories take priority
    # Remove from names any entity that appears in another category
    non_name_entities = set()
    for cat, ents in args.entities.items():
        if cat != "names":
            non_name_entities.update(ents)
    if "names" in args.entities:
        before = len(args.entities["names"])
        args.entities["names"] = [e for e in args.entities["names"] if e not in non_name_entities]
        removed = before - len(args.entities["names"])
        if removed > 0:
            print(f"  Deduplication: removed {removed} names that overlap with other categories")

    # Filter to categories with at least 4 entities (needed for sampling A, B, C, D)
    args.entities = {k: v for k, v in args.entities.items() if len(v) >= 4}

    return args


baba_templates = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

abba_templates = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to {A}",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]


def try_fit_template(string, template):
    pieces_s, pieces_t = string.strip(), template.strip()

    mapping = {}

    for s, t in zip(pieces_s.split(), pieces_t.split()):
        if s == t:
            continue
        if s[-1] == t[-1] and s[-1] in [',', '.']:
            s, t = s[:-1], t[:-1]
        if t not in ['{A}', '{B}', '{PLACE}', '{OBJECT}']:
            return None
        elif t[1:-1].lower() in mapping:
            if mapping[t[1:-1].lower()] != s:
                return None
        else:
            mapping[t[1:-1].lower()] = s

    if 'place' not in mapping:
        mapping['place'] = None
    if 'object' not in mapping:
        mapping['object'] = None

    return mapping


def find_template(string):
    for template in baba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({'template': template, 'order': 'baba'})
            return mapping

    for template in abba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({'template': template, 'order': 'abba'})
            return mapping
    return None


def add_template(example, **kwargs):
    example.update(find_template(example['ioi_sentences']))
    return example


def train_val_split_templates(data, val_templates):
    val_indices = [i for i in tqdm(range(len(data))) if data[i]['template'] in val_templates]
    train_indices = [i for i in tqdm(range(len(data))) if i not in val_indices]

    return data.select(train_indices), data.select(val_indices)


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "w")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.log.write(data)
    def flush(self):
        self.stdout.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def main():
    log_path = os.path.join(os.path.dirname(__file__), "logs", "prepare_ioi_mixed.log")
    tee = Tee(log_path)
    sys.stdout = tee

    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    categories = sorted(args.entities.keys())
    print("=" * 60)
    print("MIXED IOI DATASET GENERATION")
    print("=" * 60)
    for cat in categories:
        print(f"  {cat}: {len(args.entities[cat])} entities")

    # Part 1: Prepare the splits

    n_baba = args.num_test_templates // 2
    n_abba = args.num_test_templates - n_baba

    val_templates = random.sample(baba_templates, n_baba) + random.sample(abba_templates, n_abba)

    data = load_dataset('fahamu/ioi')['train']
    if args.start_size < len(data):
        data = data.select(range(args.start_size))
    data = data.map(add_template, num_proc=32)

    train, test = train_val_split_templates(data, val_templates)

    assert len(train) >= args.train + args.validation, f"Train size {len(train)} is too small"
    assert len(test) >= args.test, f"Test size {len(test)} is too small"

    new_train = train.select(range(args.train))
    val = train.select(range(args.train, args.train + args.validation))
    test = test.select(range(args.test))

    data = DatasetDict({
        "train": new_train,
        "validation": val,
        "test": test,
    })

    # Part 2: Replace names with mixed entities and create corrupted versions

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    processed = {}
    cat_idx = 0  # Round-robin counter across categories

    for split in data:
        processed[split] = []
        for ex in tqdm(data[split], desc=f"Processing {split}"):
            original_sentence = ex['ioi_sentences']
            original_sentence_words = original_sentence.split()

            # Pick category round-robin
            category = categories[cat_idx % len(categories)]
            cat_idx += 1
            entity_pool = args.entities[category]

            # Sample 4 entities from the same category
            while True:
                entity_a, entity_b, entity_c, entity_d = random.sample(entity_pool, 4)

                if any(e in [ex['a'], ex['b']] for e in [entity_a, entity_b, entity_c, entity_d]):
                    continue

                # Build clean sentence: both A positions -> entity_a, both B positions -> entity_b
                new_clean = original_sentence_words.copy()
                first_a = new_clean.index(ex['a'])
                first_b = new_clean.index(ex['b'])
                second_a = new_clean.index(ex['a'], first_a + 1)
                second_b = new_clean.index(ex['b'], first_b + 1)
                new_clean[first_a] = entity_a
                new_clean[first_b] = entity_b
                new_clean[second_a] = entity_a
                new_clean[second_b] = entity_b
                new_clean_str = " ".join(new_clean)

                # Build corrupted sentence: second A -> entity_c, second B -> entity_d
                new_corr = original_sentence_words.copy()
                new_corr[first_a] = entity_a
                new_corr[first_b] = entity_b
                new_corr[second_a] = entity_c
                new_corr[second_b] = entity_d
                new_corr_str = " ".join(new_corr)

                # Ensure clean and corrupted have same GPT-2 token count
                if len(tokenizer.tokenize(new_clean_str)) == len(tokenizer.tokenize(new_corr_str)):
                    break

            ex['a'] = entity_a
            ex['b'] = entity_b
            ex['c'] = entity_c
            ex['d'] = entity_d
            ex['ioi_sentences'] = new_clean_str
            ex['corr_ioi_sentences'] = new_corr_str
            ex['category'] = category
            processed[split].append(ex)

    processed = DatasetDict({
        k: Dataset.from_list(v) for k, v in processed.items()
    })

    # Print stats
    print(f"\n{'=' * 60}")
    print("DATASET STATISTICS")
    print("=" * 60)
    for split in processed:
        from collections import Counter
        cats = Counter(ex['category'] for ex in processed[split])
        orders = Counter(ex['order'] for ex in processed[split])
        print(f"  [{split}] {len(processed[split])} examples")
        print(f"    categories: {dict(sorted(cats.items()))}")
        print(f"    orders: {dict(orders)}")

    # Print examples
    print(f"\n{'=' * 60}")
    print("EXAMPLES")
    print("=" * 60)
    for split in ["train", "test"]:
        print(f"\n  [{split}]")
        for ex in list(processed[split])[:3]:
            print(f"    {ex['category']:10s} ({ex['order']}): {ex['ioi_sentences']}")
            print(f"    {'':10s}  corr: {ex['corr_ioi_sentences']}")

    # Part 3: Save
    os.makedirs(args.out_dir, exist_ok=True)
    processed.save_to_disk(args.out_dir)
    print(f"\n  Saved to {args.out_dir}")
    print(f"  Log written to {log_path}")

    sys.stdout = tee.stdout
    tee.close()


if __name__ == '__main__':
    main()
