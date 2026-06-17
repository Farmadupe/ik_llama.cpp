# Project rules

- **Write only ASCII while operating in this codebase.** Do not introduce
  non-ASCII characters (em-dashes, smart quotes, arrows, multiplication/approx
  signs, etc.) into code, comments, or docs. Use ASCII equivalents: `-` for
  em-dashes, `x` for multiplication signs, "approximately" (or `~`) for approx
  signs, plain `"`/`'` for smart quotes.
- **Do not use ASCII arrows (`->`) either.** The user dislikes them. Express the
  relationship in plain English (e.g. "from A to B", "A becomes B", "A then B")
  instead of `A -> B`. This is about prose arrows. The `->` of real language
  syntax is fine and must not be "fixed": Python return-type hints
  (`def f() -> str:`), Rust/C++ returns and member access, shell/Make rules, etc.
  The rule targets arrows used as prose shorthand, not code tokens.

# JJ
The users is using JJ for change management.