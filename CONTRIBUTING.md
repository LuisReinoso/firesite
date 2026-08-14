# Contributing

Thanks for looking. The most useful contributions right now are listed at the
bottom, but anything that makes the tool more honest or easier to run is welcome.

## Setup

```bash
git clone https://github.com/LuisReinoso/firesite
cd firesite
pip install -e ".[dev]"
pre-commit install
pytest
```

That is the whole toolchain. No compilers, no system packages, no API key needed
to run the tests.

## The one rule that shapes this codebase

**The analysis is pure functions.** A frame goes in, a frame or a plain value
comes out. Nothing mutates its input, nothing reads the network, nothing touches
disk. Only `firms.fetch_*`, `cli`, `plot` and `export.write_payload` talk to the
outside world.

This is not style preference. It is why the whole suite runs in about a second
with no fixtures, and why a FIRMS outage can never turn the build red. If you add
analysis, add it as a pure function and test it directly.

## Tests come first

Write the failing test, then the fix. The four defects found during the initial
build were all silent-in-production bugs that a test caught before a user did:
a bounding box near the poles, `top=0` meaning "everything", ties in the site
search resolved by iteration order, and an argument shim that mangled a user
error into an unreadable crash.

New behaviour needs a test. Bug fixes need a test that fails before the fix.

## Before you push

`pre-commit` runs ruff and the formatter automatically. If you would rather do it
by hand:

```bash
ruff check --fix .
ruff format .
pytest
```

CI runs the same checks on Linux, macOS and Windows across Python 3.10 to 3.13,
plus a coverage floor of 80%.

## Style

- Comments explain **why**, not what. If a line needs a comment to say what it
  does, rename something instead.
- Document limitations in the open. The README says plainly that terrain is not
  modelled and that the 8-pixel threshold is not from a paper. Keep that habit:
  a tool people trust is one that tells them where it is weak.
- Prose in docs avoids the em dash and marketing adjectives.

## Most wanted

1. **Terrain and line of sight.** The largest gap by far. Every published siting
   method is built on a digital elevation model, and firesite ignores terrain
   entirely, so a highly ranked position may sit behind a ridge. See
   `docs/literature.md`.
2. **Multi-camera placement.** The literature frames this as a covering location
   problem, where the best pair of sites is not the two best individual ones.
   The current search cannot express that.
3. **Fire histories from other regions** to test the persistent-source filter
   against. It was tuned on Andean data and has not been checked against boreal,
   Mediterranean or savanna fire regimes.
4. **A citation for the pixels-on-target threshold**, or evidence that it is
   wrong.
