# AGENTS.md

## Python conventions

- Target Python 3.13 and add complete type hints to functions.
- Use modern built-in annotations such as `str | Path` and
  `dict[str, float]`.
- Order imports: standard library, third party, then local package imports.
- Functions and methods need concise Google-style docstrings covering
  rationale, arguments, returns, and raised exceptions when applicable.
  Example:
  ```python
  def add(a: int, b: int) -> int:
      """Add two integers.

      Args:
          a: The first integer.
          b: The second integer.

      Returns:
          The sum of the two integers.

      Raises:
          ValueError: If either argument is not an integer.
      """
      if not isinstance(a, int) or not isinstance(b, int):
          raise ValueError("Both arguments must be integers.")
      return a + b
  ```
  Note that the type hints in the docstring are omitted because they are already
  specified in the function signature.
- Prefer focused functions over classes; use a dataclass only where state is
  genuinely required.
- Use descriptive `snake_case` names and `UPPER_SNAKE_CASE` constants.
  Prefix private helpers and module constants with `_`.
- Keep comments rare: explain non-obvious rationale, not mechanics.
- Fail explicitly on invalid or unavailable data. Preserve existing logging and
  exception patterns; do not add silent fallbacks or broad exception handling.
- Use absolute imports for local packages rather than relative imports.
- Do not check for rare edge cases; focus on common and likely scenarios. For rare
  cases, it is okay to let them fail naturally rather than adding complex handling.
- No magic numbers: avoid using unexplained literal values in the code; define them
  as named constants instead.
- Keep __init__.py files blank unless they need to perform package initialization.
- Function and method names should start with a verb.
- Prefer early returns to reduce nested code and improve readability.
- Put public functions at the top of the module and private helpers below them, so
  that the module's interface is immediately clear to readers.
