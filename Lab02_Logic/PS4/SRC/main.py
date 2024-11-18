import os
from knowledge_base import Clause, KnowledgeBase
from utils import read_input, write_output, pl_resolution, data_structuring

def process_single_file(input_path: str, output_path: str) -> None:
    """Process a single input file and write results to output file"""
    # Read input
    lines = read_input(input_path)
    if not lines:
        print(f"Error reading input file: {input_path}")
        return
        
    # Process input
    alpha, kb_strings = data_structuring(lines)
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    for clause_str in kb_strings:
        kb.add_clause(Clause.from_string(" OR ".join(clause_str)))
    
    # Run resolution
    entails, steps = pl_resolution(kb, alpha)
    
    # Write output
    write_output(output_path, entails, steps, kb)
    print(f"Processed: {input_path} -> {output_path}")

def main():
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .txt files from input directory
    try:
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found")
        return
    
    if not input_files:
        print(f"No .txt files found in {input_dir} directory")
        return
    
    # Process each input file
    for input_file in sorted(input_files):
        input_path = os.path.join(input_dir, input_file)
        # input1.txt -> output1.txt
        output_file = input_file.replace("input", "output")
        output_path = os.path.join(output_dir, output_file)
        
        try:
            process_single_file(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    main()