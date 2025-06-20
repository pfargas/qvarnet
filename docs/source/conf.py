# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the source directory to Python path
# With docs/source/ structure, we need to go up two levels to reach src/
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
project = 'qvarnet'
copyright = '2025, Pau Fargas'
author = 'Pau Fargas'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
    'sticky_navigation': True,
    'titles_only': False,
    'includehidden': True,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MathJax configuration for quantum mechanics notation
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'ket': [r'\left|#1\right\rangle', 1],
            'bra': [r'\left\langle#1\right|', 1],
            'braket': [r'\left\langle#1\middle|#2\right\rangle', 2],
            'expval': [r'\left\langle#1\right\rangle', 1],
            'psi': r'\psi',
            'Psi': r'\Psi',
            'hamiltonian': r'\hat{H}',
            'hbar': r'\hbar',
        }
    }
}

# Add debugging print to verify path
print(f"Python path for Sphinx: {sys.path[0]}")
print(f"Current working directory: {os.getcwd()}")
print(f"Source path exists: {os.path.exists(os.path.abspath('../../src'))}")
print(f"Source contents: {os.listdir(os.path.abspath('../../src')) if os.path.exists(os.path.abspath('../../src')) else 'Not found'}")