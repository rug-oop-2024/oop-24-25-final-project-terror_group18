import os
import subprocess

# Define the directories to generate documentation for
directories = ["app", "autoop"]

# Create a folder to store the generated docs
os.makedirs("docs", exist_ok=True)

# Function to generate documentation for a given .py file
def generate_pydoc(file_path, output_dir="docs"):
    # Extract the module path (e.g., app.pages.Welcome) from the file path
    module_path = file_path.replace(os.sep, ".").replace(".py", "")
    output_file = os.path.join(output_dir, f"{module_path}.html")

    # Run `pydoc -w` to generate HTML documentation
    print(f"Generating documentation for {module_path}")
    subprocess.run(["pydoc", "-w", module_path])

    # Move the generated file to the docs directory
    generated_file = f"{module_path}.html"
    if os.path.exists(generated_file):
        os.rename(generated_file, os.path.join(output_dir, generated_file))

# Iterate over each directory and generate documentation for all .py files
for directory in directories:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                generate_pydoc(file_path)

print("Documentation generation complete!")

