Contributing
============

We welcome contributions to Bayes-HDC.

Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/bayes-hdc.git
   cd bayes-hdc

2. Create a virtual environment and install:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"

Code Style
----------

We use ``ruff`` for linting and formatting:

.. code-block:: bash

   ruff check bayes_hdc/ tests/
   ruff format bayes_hdc/ tests/

Type checking:

.. code-block:: bash

   mypy bayes_hdc/

Testing
-------

.. code-block:: bash

   pytest tests/ -v

Submitting Changes
------------------

1. Create a feature branch
2. Make changes and add tests
3. Ensure all tests pass and code is formatted
4. Submit a pull request
