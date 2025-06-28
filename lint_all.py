import os
from pylint import lint
import black
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
files = [i for i in os.listdir(current_dir) if i.endswith(".py")]

for _, file in enumerate(files):
    print(file)
    print(f"Cleaning file {_+1} out of {len(files)}")
    black.format_file_in_place(
        black.Path(file),
        fast=False,
        mode=black.FileMode(),
        write_back=black.WriteBack.YES,
    )
    print("File Clean!")
    print(f"Checking file {_+1} out of {len(files)} for further errors...")
    subprocess.run(["pylint", file])
    continue
