"""
Script to add "Điều {lesson_number}" prefix to chunk text
"""
import json
import os
from pathlib import Path


def add_dieu_to_chunks(input_file, output_file=None):
    """
    Read lessons from JSON file and add "Điều {lesson_number}" to the beginning of each chunk's text.
    
    Args:
        input_file: Path to input JSON file containing lessons
        output_file: Path to output JSON file. If None, creates a new file with '_dieu' suffix
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lessons = json.load(f)
    
    # Process each lesson
    for lesson in lessons:
        lesson_num = lesson.get('lesson')
        if lesson_num is not None:
            dieu_prefix = f"Điều {lesson_num} "
            
            # Add prefix to each chunk's text
            if 'chunks' in lesson:
                for chunk in lesson['chunks']:
                    original_text = chunk.get('text', '')
                    # Only add prefix if text doesn't already start with "Điều"
                    if original_text and not original_text.startswith('Điều'):
                        chunk['text'] = dieu_prefix + original_text
    
    # Determine output file path
    if output_file is None:
        base_name = Path(input_file).stem
        output_file = os.path.join(
            os.path.dirname(input_file),
            f"{base_name}_with_dieu.json"
        )
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Processed {len(lessons)} lessons")
    print(f"✓ Output saved to: {output_file}")
    
    return output_file


def process_all_lesson_files(data_dir='data/lesson', output_dir='penalty'):
    """
    Process all lesson JSON files in a directory.
    
    Args:
        data_dir: Directory containing lesson JSON files
        output_dir: Directory to save processed files
    """
    lesson_dir = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all JSON files in lesson directory
    json_files = list(lesson_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} lesson files\n")
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        output_file = output_path / f"{json_file.stem}_with_dieu.json"
        add_dieu_to_chunks(str(json_file), str(output_file))
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        add_dieu_to_chunks(input_file, output_file)
    else:
        # Process all lesson files
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        data_dir = os.path.join(parent_dir, 'data', 'lesson')
        output_dir = current_dir
        
        process_all_lesson_files(data_dir, output_dir)
