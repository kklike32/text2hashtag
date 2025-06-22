#!/usr/bin/env python3
"""
Creates the Conda environment from environment.yml.
"""
import subprocess

def main():
    subprocess.run(
        ["conda", "env", "create", "--file", "environment.yml"],
        check=True
    )

if __name__ == "__main__":
    main()

