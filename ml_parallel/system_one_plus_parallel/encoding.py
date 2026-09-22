"""A shared state, isolated questions, and isolated candidates within each question."""

from dataclasses import dataclass

# Existing Qwen tokens: no embedding resize and no trainable vocabulary head.
TOKENS = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "context": "<|fim_suffix|>",
    "candidate": "<|box_start|>",
    "candidate_end": "<|box_end|>",
    "propose": "<|quad_start|>",
    "action": "<|object_ref_start|>",
    "action_end": "<|object_ref_end|>",
    "done": "<|quad_end|>",
}
ENCODING_VERSION = 2


@dataclass(frozen=True)
class Encoding:
    prefix: tuple[int, ...]
    candidates: tuple[tuple[int, ...], ...] = ()
    proposal: tuple[int, ...] = ()
    state_length: int = 0

    @property
    def size(self):
        return len(self.prefix) + sum(map(len, self.candidates)) + len(self.proposal)

    @property
    def next_position(self):
        return len(self.prefix) + max(map(len, self.candidates), default=0) + len(self.proposal)


@dataclass(frozen=True)
class ParallelEncoding:
    state: tuple[int, ...]
    questions: dict[str, Encoding]

    def __post_init__(self):
        if not self.state or not self.questions:
            raise ValueError("a parallel encoding needs a state and at least one question")
        for key, question in self.questions.items():
            if not key.strip() or question.prefix[: len(self.state)] != self.state:
                raise ValueError("questions must have IDs and share exactly the same state prefix")
            if question.state_length != len(self.state):
                raise ValueError("question state boundary does not match shared state")
            if len(question.prefix) <= len(self.state) or any(not c for c in question.candidates):
                raise ValueError("empty question or candidate branch")

    @property
    def size(self):
        return len(self.state) + sum(q.size - len(self.state) for q in self.questions.values())


class Codec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = {}
        for name, text in TOKENS.items():
            token = tokenizer.convert_tokens_to_ids(text)
            if token is None or token == tokenizer.unk_token_id:
                raise ValueError(f"missing protocol token: {text}")
            if tokenizer.encode(text, add_special_tokens=False) != [token]:
                raise ValueError(f"protocol token must encode as one ID: {text}")
            self.ids[name] = token
        if len(set(self.ids.values())) != len(TOKENS):
            raise ValueError("protocol token IDs must be distinct")
        self.reserved = set(tokenizer.all_special_ids) | set(self.ids.values())

    def text(self, text):
        # Escape the complete reserved spelling, including pretrained chat tokens.
        for spelling in sorted(
            set(self.tokenizer.all_special_tokens) | set(TOKENS.values()), key=len, reverse=True
        ):
            text = text.replace(spelling, spelling.replace("<", "‹").replace(">", "›"))
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        # Some minimal tokenizers use UNK for ordinary words.
        forbidden = self.reserved - {self.tokenizer.unk_token_id}
        if forbidden.intersection(tokens):
            raise ValueError("ordinary text contains a reserved control ID")
        return tokens

    def candidate(self, text):
        if not text.strip():
            raise ValueError("candidate text must be nonempty")
        return (self.ids["candidate"], *self.text(text), self.ids["candidate_end"])

    def encode(
        self,
        state,
        question,
        choices=(),
        *,
        targets=None,
        max_state=384,
        max_length=1024,
        max_work_tokens=4096,
        max_cache_tokens=4096,
    ):
        if not state.strip() or not question.strip():
            raise ValueError("state and question must be nonempty")
        state_ids = self.text(state)
        if len(state_ids) + 1 > max_state:
            raise ValueError("state exceeds max_state_tokens; silent truncation is disabled")
        prefix = (
            self.ids["state"],
            *state_ids,
            self.ids["question"],
            *self.text(question),
            self.ids["context"],
        )
        proposal = []
        if targets is not None:
            proposal = [self.ids["propose"]]
            for action in targets:
                if not action.strip():
                    raise ValueError("proposal target must be nonempty")
                proposal.extend([self.ids["action"], *self.text(action), self.ids["action_end"]])
            proposal.append(self.ids["done"])
        result = Encoding(
            tuple(prefix), tuple(self.candidate(c) for c in choices), tuple(proposal), len(state_ids) + 1
        )
        if result.next_position > max_length:
            raise ValueError(f"branch context exceeds max_length={max_length}")
        if result.size > max_work_tokens:
            raise ValueError(f"input has {result.size} tokens, exceeding total-work budget={max_work_tokens}")
        if proposal and result.size > max_cache_tokens:
            raise ValueError("teacher proposal exceeds proposal cache capacity")
        return result

    def encode_questions(
        self,
        state,
        questions,
        *,
        max_state=384,
        max_length=1024,
        max_packed_tokens=4096,
        max_cache_tokens=4096,
    ):
        if not questions or len(questions) > 64:
            raise ValueError("provide between 1 and 64 questions")
        encoded = {
            key: self.encode(
                state,
                item["instructions"],
                item.get("choices", {}).values(),
                targets=item.get("targets"),
                max_state=max_state,
                max_length=max_length,
                max_work_tokens=max_packed_tokens,
                max_cache_tokens=max_cache_tokens,
            )
            for key, item in questions.items()
        }
        first = next(iter(encoded.values()))
        state_end = first.state_length
        packed = ParallelEncoding(first.prefix[:state_end], encoded)
        if packed.size > max_packed_tokens:
            raise ValueError(f"packed request has {packed.size} tokens, exceeding {max_packed_tokens}")
        return packed
