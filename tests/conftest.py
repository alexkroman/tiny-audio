"""Pytest configuration and fixtures."""

import os

# Disable tokenizers parallelism to avoid fork warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"
