"""Hugging Face Spaces entry point.

The Gradio SDK expects a root-level `app.py` exposing a Gradio Blocks
instance named `demo`. The actual UI lives in src/demo/app.py, importable
and testable on its own; this file just re-exports it for Spaces.

Settings to set in the Spaces UI when creating the Space (not in code):
- SDK: Gradio
- SDK version: 6.20.0 (matches src/demo/requirements.txt)
- Hardware: CPU basic — no GPU needed, everything here runs on CPU
- Visibility: public or private, your choice
"""

from src.demo.app import demo

if __name__ == "__main__":
    demo.launch()
