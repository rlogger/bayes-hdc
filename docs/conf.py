# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""Sphinx configuration for the bayes-hdc documentation site.

Builds with the Furo theme (modern, dark-mode-aware) and pulls the
project version from ``bayes_hdc.__version__`` so the docs and the
package always agree.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

# ----------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------

project = "bayes-hdc"
author = "Rajdeep Singh"
copyright = f"{datetime.now().year}, {author}"

try:  # Pull the version from the installed package so docs always match.
    from bayes_hdc import __version__ as _pkg_version
    release = _pkg_version
    version = ".".join(_pkg_version.split(".")[:2])
except Exception:  # pragma: no cover — docs build before package importable
    release = "0.4.0a0"
    version = "0.4"

# ----------------------------------------------------------------------
# Extensions
# ----------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "smartquotes",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ----------------------------------------------------------------------
# Build behaviour
# ----------------------------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Per-paper audit artefacts (readable on GitHub, not part of the docs site).
    "LITERATURE_AUDIT.md",
    "audit/*",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# Treat warnings as errors when CI sets BAYES_HDC_DOCS_STRICT=1.
nitpicky = bool(int(os.environ.get("BAYES_HDC_DOCS_STRICT", "0")))

# ----------------------------------------------------------------------
# HTML output (Furo theme)
# ----------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "bayes-hdc"
html_short_title = "bayes-hdc"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False
html_show_sphinx = False

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/rlogger/bayes-hdc/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/rlogger/bayes-hdc",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-'
                '.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
                '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52'
                '.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2'
                '-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09'
                ' 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82'
                ' 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01'
                ' 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'
                "</svg>"
            ),
            "class": "",
        },
    ],
    "light_css_variables": {
        "color-brand-primary": "#1e1e3f",
        "color-brand-content": "#1e1e3f",
        "color-admonition-background": "#fafafc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff9f6b",
        "color-brand-content": "#ff9f6b",
    },
}

# Code-copy button styling.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ----------------------------------------------------------------------
# SEO and social-card metadata
# ----------------------------------------------------------------------

# Canonical site URL for sitemap.xml + OG tags.
html_baseurl = "https://rlogger.github.io/bayes-hdc/"

# sphinx_sitemap config: emit sitemap.xml at the site root.
sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"

# OpenGraph + Twitter card. The image is generated by Sphinx into
# _static/social-card.png at build time (see docs/Makefile-equivalent
# helper) — we point at the SVG fallback so social shares always work.
ogp_site_url = html_baseurl
ogp_site_name = "bayes-hdc — Probabilistic hyperdimensional computing in JAX"
ogp_image = f"{html_baseurl}_static/social-card.svg"
ogp_image_alt = (
    "bayes-hdc logo and tagline: probabilistic hyperdimensional computing in JAX"
)
ogp_description_length = 240
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image"/>',
    '<meta name="twitter:title" content="bayes-hdc — Probabilistic hyperdimensional computing in JAX"/>',
    '<meta name="twitter:description" content="A JAX library for hyperdimensional computing with closed-form Gaussian and Dirichlet hypervectors, group-theoretic equivariance, calibrated probabilities, and coverage-guaranteed conformal prediction sets."/>',
    f'<meta name="twitter:image" content="{html_baseurl}_static/social-card.svg"/>',
    # Page-level keywords. Search-engine signal, not user-facing.
    '<meta name="keywords" content="hyperdimensional computing, vector symbolic architectures, VSA, HDC, JAX, Bayesian, probabilistic, conformal prediction, calibration, temperature scaling, Gaussian hypervector, Dirichlet hypervector, neuromorphic, edge ML, Kanerva, HRR, BSC, MAP, Hopfield, sparse distributed memory"/>',
    '<meta name="author" content="Rajdeep Singh"/>',
    '<meta name="robots" content="index, follow"/>',
    # Schema.org JSON-LD for the software application.
    """<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "bayes-hdc",
  "alternateName": "Bayes-HDC",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Linux, macOS",
  "softwareVersion": "0.4.0a0",
  "programmingLanguage": "Python",
  "description": "A JAX library for probabilistic hyperdimensional computing with closed-form Gaussian and Dirichlet hypervectors, group-theoretic equivariance, calibrated probabilities, and coverage-guaranteed conformal prediction sets.",
  "license": "https://opensource.org/licenses/MIT",
  "codeRepository": "https://github.com/rlogger/bayes-hdc",
  "url": "https://rlogger.github.io/bayes-hdc/",
  "author": {"@type": "Person", "name": "Rajdeep Singh"},
  "keywords": "hyperdimensional computing, vector symbolic architectures, JAX, Bayesian machine learning, conformal prediction, uncertainty quantification"
}
</script>""",
]

# Robots.txt — allow all crawlers, point at sitemap.
html_extra_path = ["_extra"]
