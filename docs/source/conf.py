# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
import sphinx_theme_pd

project = "Open MatSciML Toolkit"
copyright = "2024, Intel Corporation"
author = "Intel Corporation"

extensions = ["sphinxawesome_theme.highlighting"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinxawesome_theme"
html_theme_path = [sphinx_theme_pd.get_html_theme_path()]
html_static_path = ["_static"]
