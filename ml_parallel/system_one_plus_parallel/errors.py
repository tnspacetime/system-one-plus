"""Separate caller validation from model/decoding failures."""


class InvalidRequestError(ValueError):
    pass


class ProposalOutputError(RuntimeError):
    def __init__(self, message, *, status="generation_failed"):
        super().__init__(message)
        self.status = status
