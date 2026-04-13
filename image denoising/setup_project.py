"""
Run this FIRST to create the project folder structure.
Command: python setup_project.py
"""
import os

folders = [
    "data/noisy",
    "data/clean",
    "data/output",
    "modules",
    "noise_maps",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}/")

# Create __init__.py so modules folder works as a package
with open("modules/__init__.py", "w") as f:
    f.write("")

print("\nProject structure ready!")
print("Now run: streamlit run app.py")