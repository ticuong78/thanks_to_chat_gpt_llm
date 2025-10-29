# Ollama singletons

import ollama

from .argument.arguments import host

ollama_client = ollama.Client(host=host)

__all__ = ("ollama_client",)