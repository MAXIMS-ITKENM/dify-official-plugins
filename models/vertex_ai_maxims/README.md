# Vertex AI Maxims Model Plugin

This directory contains the implementation for the Vertex AI Maxims model plugin for Dify.

## Features
- Google Vertex AI LLM integration
- Caching of authenticated clients using TTLCache
- Support for custom labels, tools, and advanced configuration
- Modular, maintainable code structure

## Requirements
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (for fast dependency installation)
- Google Cloud credentials (service account)

## Installation

1. **Clone the repository** (if you haven't already):
   ```sh
   git clone <your-repo-url>
   cd dify-official-plugins/models/vertex_ai_maxims
   ```

2. **(Recommended) Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   # On Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies using uv:**
   ```sh
   uv pip install -r requirements.txt
   ```

## Usage

- The main entry point is the `VertexAiLargeLanguageModel` class in `models/llm/llm.py`.
- Integrate this model with your Dify plugin system as needed.
- Example usage:

   ```python
   from models.llm.llm import VertexAiLargeLanguageModel
   # ...initialize with model schemas and use as needed...
   ```

## Development
- Code is structured for maintainability and extensibility.
- See `models/llm/llm.py` for main logic and helper methods.
- Caching is handled with `cachetools.TTLCache` for efficient client reuse.

## Notes
- Ensure your Google Cloud service account has the necessary permissions for Vertex AI.
- For more details on the Dify plugin SDK, see the [Dify Plugin SDK DeepWiki](https://deepwiki.com/langgenius/dify-plugin-sdks).
- For more on the Google GenAI Python SDK, see the [Google GenAI DeepWiki](https://deepwiki.com/googleapis/python-genai).

## License
See the root `LICENSE` file for license information.
