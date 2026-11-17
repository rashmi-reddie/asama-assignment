from hw_assistant.runner import SourceResult
from hw_assistant.tools import all_sources


class FakeRunner:
    """Returns canned raw text keyed by parser name; records which sources were read."""

    def __init__(self, raw_by_parser: dict, fail_parsers: set | None = None):
        self.raw, self.fail, self.reads = raw_by_parser, fail_parsers or set(), []

    def read(self, src):
        self.reads.append(src.parser)
        if src.parser in self.fail or src.parser not in self.raw:
            return SourceResult(False, error="not available", error_kind="missing", source=src.describe())
        return SourceResult(True, raw=self.raw[src.parser], source=src.describe())
