import os
import sys
import sphinx_rtd_theme

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyRootMancer'
copyright = '2024, Jakub Cyba, Simona Dimitrova, Cedric Verhaegh, Thomas Picardo, Samuel'
author = 'Jakub Cyba, Simona Dimitrova, Cedric Verhaegh, Thomas Picardo, Samuel'
release = '0.1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
# Logo settings
# Adjust the path to your logo file
html_theme_options = {
    'logo_only': True,
    'navigation_with_keys': True,
}
