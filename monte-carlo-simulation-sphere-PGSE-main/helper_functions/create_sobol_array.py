#!/home/software/python/3.9.4/bin/python3

import sys
import argparse
import os  
import numpy as np  
from SALib.sample import sobol

def create_sobol_samples(file_path):
    """
    Reads a parameter file, defines a SALib problem, and generates Sobol samples.
    (This function remains unchanged from the previous version)
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.", file=sys.stderr)
        return None

    if not lines:
        print("Error: The file is empty.", file=sys.stderr)
        return None

    try:
        n_value = int(lines[-1])
        bounds = []
        for line in lines[:-1]:
            parts = line.split()
            lower = float(parts[0])
            upper = float(parts[1])
            bounds.append([lower, upper])
    except (ValueError, IndexError) as e:
        print(f"Error: Could not parse file. Check for non-numeric values or incorrect formatting.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        return None

    num_vars = len(bounds)
    names = [f'x{i+1}' for i in range(num_vars)]

    problem = {
        'num_vars': num_vars,
        'names': names,
        'bounds': bounds
    }

    print("--- Problem Definition ---")
    print(f"Sample Size (N): {n_value}")
    print(f"Number of Variables: {num_vars}")
    print(f"Bounds: {problem['bounds']}")
    print("--------------------------\n")
    
    n_value_in = int(np.round(n_value/(num_vars+2)))
    print(f"Power of 2: {n_value_in}")
    param_values = sobol.sample(problem, n_value_in, calc_second_order=False)

    return param_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Sobol samples from a parameter file and save the output.",
        epilog="Example: python create_sobol_array.py my_params.txt"
    )
    parser.add_argument("file_path", help="The path to the input parameter file.")
    
    args = parser.parse_args()
    
    parameter_array = create_sobol_samples(args.file_path)
    
    # Check if the array was created successfully before trying to save
    if parameter_array is not None:
        print(f"Successfully generated parameter array with shape: {parameter_array.shape}")

        try:
            output_filename = "sobol_array.txt"
            
            input_directory = os.path.dirname(args.file_path)
            
            output_path = os.path.join(input_directory, output_filename)
            
            np.savetxt(output_path, parameter_array, fmt='%.8f', delimiter=' ')
            
            print(f"\nArray successfully saved to:")
            print(os.path.abspath(output_path))

        except Exception as e:
            print(f"\nError: Could not save the output file.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
