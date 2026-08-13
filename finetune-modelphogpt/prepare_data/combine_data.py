#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine lesson and dialogue fine-tuning JSONL files into a single training file.

Source folders (expected existing):
  - dataset/lesson/*.jsonl
  - dataset/dialogue/*.jsonl

Output:
  - dataset/train.jsonl (merged, deduplicated by prompt text)

Rules:
  - Each input line must be a JSON object with keys: prompt, response.
  - Skip lines missing these keys or empty values.
  - Deduplicate by exact prompt string (first occurrence wins).
  - Preserve original ordering: all lesson files (sorted by name) first, then dialogue files (sorted by name).
  - UTF-8, ensure_ascii=False.

Usage:
	python3 prepare_data/combine_data.py --out dataset/train.jsonl

Optional flags:
	--lesson-dir dataset/lesson --dialogue-dir dataset/dialogue

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def iter_jsonl(path: Path) -> Iterator[Dict]:
	with path.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			yield obj


def collect_files(folder: Path) -> List[Path]:
	if not folder.exists():
		return []
	return sorted([p for p in folder.glob('*.jsonl') if p.is_file()])


def merge_datasets(lesson_files: List[Path], dialogue_files: List[Path]) -> Iterator[Dict]:
	"""Merge lesson and dialogue files without deduplicating prompts.

	All valid lines from all files are kept. Lessons are yielded first,
	then dialogues, but duplicates of the same prompt are no longer removed.
	"""

	def emit(files: List[Path]):
		for fp in files:
			for obj in iter_jsonl(fp):
				prompt = obj.get('prompt')
				response = obj.get('response')
				if not prompt or not response:
					continue
				yield {'prompt': prompt, 'response': response}

	# Lessons first, then dialogues
	yield from emit(lesson_files)
	yield from emit(dialogue_files)


def write_jsonl(objs: Iterable[Dict], out_path: Path) -> int:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	n = 0
	with out_path.open('w', encoding='utf-8') as f:
		for obj in objs:
			f.write(json.dumps(obj, ensure_ascii=False) + '\n')
			n += 1
	return n


def main():
	ap = argparse.ArgumentParser(description='Combine lesson & dialogue JSONL into one training file.')
	ap.add_argument('--lesson-dir', default='dataset/lesson', help='Directory containing lesson JSONL files')
	ap.add_argument('--dialogue-dir', default='dataset/dialogue', help='Directory containing dialogue JSONL files')
	ap.add_argument('--out', default='dataset/train.jsonl', help='Output JSONL path')
	args = ap.parse_args()

	lesson_dir = Path(args.lesson_dir)
	dialogue_dir = Path(args.dialogue_dir)
	out_path = Path(args.out)

	lesson_files = collect_files(lesson_dir)
	dialogue_files = collect_files(dialogue_dir)

	merged_iter = merge_datasets(lesson_files, dialogue_files)
	total = write_jsonl(merged_iter, out_path)
	print(f'✅ Combined {len(lesson_files)} lesson file(s) and {len(dialogue_files)} dialogue file(s) -> {total} records at {out_path}')


if __name__ == '__main__':
	main()

