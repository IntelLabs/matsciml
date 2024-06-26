project = "Open MatSciML Toolkit"
copyright = "2024, Intel Corporation"
author = "Intel Corporation"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]

# configure napoleon docstrings
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
