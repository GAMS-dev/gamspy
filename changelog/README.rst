This directory contains "newsfragments" which are short files that contain a small **ReST**-formatted
text that will be added to the next ``CHANGELOG``.

The ``CHANGELOG`` will be read by **users** and **developers**.

Each file should be named like ``<ISSUE>.<TYPE>.rst``, where
``<ISSUE>`` is an issue number, and ``<TYPE>`` is one of:

* ``feature``: new user facing features, like new command-line options and new behavior.
* ``improvement``: improvement of an existing functionality.
* ``bugfix``: fixes a bug.
* ``doc``: documentation changes.
* ``deprecation``: feature deprecation.
* ``dependency``: changes in dependencies (version update etc.).
* ``cicd``: ci/cd related changes.
* ``misc``: changes that are hard to assign to any of the above
  categories.

So for example: ``123.feature.rst``, ``456.bugfix.rst``.

.. tip::

   See :file:`pyproject.toml` for all available categories
   (``tool.towncrier.type``).

If your PR fixes an issue, use that number here. If there is no issue,
then after you submit the PR and get the PR number you can add a
changelog using that instead.

If you are not sure what issue type to use, don't hesitate to ask in your PR.

``towncrier`` preserves multiple paragraphs and formatting (code blocks, lists, and so on), but for entries
other than ``features`` it is usually better to stick to a single paragraph to keep it concise.
