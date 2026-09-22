"""Reviewed paired scenarios for joint System One+ training."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    n_generate: int = Field(default=3, ge=1, le=32)
    proposal_branches: int = Field(default=1, ge=1, le=32)
    temperature: float = Field(default=0.0, ge=0, le=5)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_search(self):
        if self.proposal_branches > self.n_generate:
            raise ValueError("proposal_branches cannot exceed n_generate")
        if self.proposal_branches > 1 and self.temperature == 0:
            raise ValueError("multiple proposal branches require sampling (temperature > 0)")
        return self


class QuestionRequest(GenerationOptions):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    instructions: str = Field(min_length=1)
    mode: Literal["choose", "challenge", "suggest"] = "choose"
    choices: dict[str, str] = Field(default_factory=dict, max_length=255)
    top_k: int | None = Field(default=None, ge=1, le=255)

    @model_validator(mode="after")
    def validate_menu(self):
        if self.mode == "suggest" and self.choices:
            raise ValueError("suggest takes no choices; use challenge for a supplied menu")
        if self.mode != "suggest" and not self.choices:
            raise ValueError("choose/challenge requires choices")
        if any(not key.strip() or not value.strip() for key, value in self.choices.items()):
            raise ValueError("choice IDs and text must be nonempty")
        if len({normalized(v) for v in self.choices.values()}) != len(self.choices):
            raise ValueError("supplied choices contain duplicate text")
        return self


class ParallelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    state: str = Field(min_length=1)
    questions: dict[str, QuestionRequest] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_ids(self):
        if any(not key.strip() for key in self.questions):
            raise ValueError("question IDs must be nonempty")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    complete_choices: list[str] = Field(min_length=2, max_length=255)
    best_choice: str = Field(min_length=1)
    incomplete_choices: list[str] = Field(min_length=1, max_length=254)
    rejected_proposals: list[str] = Field(default_factory=list, max_length=8)
    acceptable_equivalents: list[str] = Field(default_factory=list)
    rubric: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_choices(self):
        fields = (
            "complete_choices",
            "incomplete_choices",
            "rejected_proposals",
            "acceptable_equivalents",
        )
        for name in fields:
            values = getattr(self, name)
            keys = [normalized(value) for value in values]
            if any(not key for key in keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} contains empty or duplicate actions")
        if self.best_choice not in self.complete_choices:
            raise ValueError("best_choice must exactly match a complete choice")
        complete = {normalized(value) for value in self.complete_choices}
        incomplete = {normalized(value) for value in self.incomplete_choices}
        accepted = {
            normalized(self.best_choice),
            *(normalized(value) for value in self.acceptable_equivalents),
        }
        if not incomplete < complete or incomplete & accepted:
            raise ValueError("incomplete_choices must omit every accepted best action")
        if complete & (accepted - {normalized(self.best_choice)}):
            raise ValueError("put accepted paraphrases in acceptable_equivalents, not the menu")
        if {normalized(value) for value in self.rejected_proposals} & (complete | accepted):
            raise ValueError("rejected proposals must be distinct from menu and accepted actions")
        return self
