import subprocess
import os

def run_segmentation(input_dir, output_dir):
    """Runs the segmentation_algo.py script with the provided input and output directories."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the segmentation algorithm script with explicit arguments
        subprocess.run(['python', 'segmentation_algo.py', '--input_dir', input_dir, '--output_dir', output_dir], check=True)
        print(f"Segmentation complete. Output saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running segmentation_algo.py: {e}")
        exit(1)

def run_epithelium_patches(input_dir, output_dir):
    """Runs the epithelium_patches_6_hori.py script with the provided input and output directories."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the epithelium patches script with the updated input_dir
        subprocess.run(['python', 'epithelium_patches_6_hori.py', '--input_folder', input_dir, '--output_folder', output_dir], check=True)
        print(f"Epithelium patches complete. Output saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running epithelium_patches_6_hori.py: {e}")
        exit(1)

def main():
    # Step 1: Get user input for directory names
    input_dir = input("Your inputs for the segmentation algorithm should be in a sub folder of the data folder. Insert your input folder path here: ").strip()
    if not os.path.isdir(input_dir):
        print(f"The input directory '{input_dir}' does not exist.")
        return
    
    output_dir = input("Outputs from epithelium separation will be stored in an intermediate folder inside of the extracted subfolder. Specify the epithelium extraction output folder here: ").strip()
    if not os.path.isdir(output_dir):
        print(f"The output directory '{output_dir}' does not exist. Creating it.")
        os.makedirs(output_dir)
    
    # Step 2: Run segmentation_algo.py
    run_segmentation(input_dir, output_dir)

    # Step 3: Get user input for final output folder
    final_output_dir = input("Final outputs from the patching algorithm will be stored in a new folder. Specify final output folder here: ").strip()
    if not os.path.isdir(final_output_dir):
        print(f"The final output directory '{final_output_dir}' does not exist. Creating it.")
        os.makedirs(final_output_dir)

    # Step 4: Run epithelium_patches_6_hori.py with the output_dir from segmentation
    run_epithelium_patches(output_dir, final_output_dir)

if __name__ == '__main__':
    main()
