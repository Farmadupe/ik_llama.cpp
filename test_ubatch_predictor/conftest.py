"""Pytest wiring for the ubatch-predictor black-box suite.

Adds --emit-json, which dumps the per-row prediction results (expected, predicted,
tolerance, acceptance bounds, relative error, regime, status) as JSON. The dump is
written from the terminal-summary hook, so it bypasses pytest's stdout capture and needs
no -s flag:

    ./run.sh --emit-json            # pretty JSON to stdout at end of run
    ./run.sh --emit-json=out.json   # clean JSON file, e.g. for jq

The test appends one object per trace to config._ubatch_json_sink (exposed as the
json_sink fixture); this hook serializes the collected list once the run finishes.
"""

import json

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("ubatch-predictor")
    group.addoption(
        "--emit-json",
        dest="emit_json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="Emit per-row predictor results as JSON. Bare or '-' writes to stdout at the "
        "end of the run; PATH writes a file.",
    )


def pytest_configure(config):
    # A list collects records only when the flag is present; otherwise None, and the test
    # skips building records at all.
    config._ubatch_json_sink = [] if config.getoption("emit_json") is not None else None


@pytest.fixture(scope="session")
def json_sink(request):
    return request.config._ubatch_json_sink


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    sink = getattr(config, "_ubatch_json_sink", None)
    if not sink:
        return
    payload = json.dumps(sink, indent=2)
    dest = config.getoption("emit_json")
    if dest in (None, "-"):
        terminalreporter.write_line(payload)
    else:
        with open(dest, "w") as handle:
            handle.write(payload + "\n")
        terminalreporter.write_line(f"wrote predictor JSON to {dest}")
