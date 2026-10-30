# SPDX-License-Identifier: Apache-2.0
"""Stage 1: paragraph-conditioned synthetic questions with explicit split assignment."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable

from .config import PAPER_QA_MODEL
from .corpus import Document


def generate_queries(
    document: Document,
    generator: Callable[[str, int], list[str]],
    *,
    questions_per_paragraph: int = 6,
    seed: int = 42,
    train_fraction: float = 0.33,
    dev_fraction: float = 0.12,
) -> list[dict]:
    """Generate questions from every leaf paragraph; stratify held-out splits by leaf.

    Ratios/seed/question count are configurable engineering choices, not constants
    supplied by the paper. Callbacks receive source text, not a target section name.
    """
    if questions_per_paragraph < 3:
        raise ValueError("Generate at least 3 questions per paragraph for train/dev/test")
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 or train_fraction + dev_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave room for test data")
    rng, seen, rows = random.Random(seed), set(), []
    for leaf in document.leaves:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", leaf.text) if p.strip()]
        questions = []
        for paragraph in paragraphs:
            generated = generator(paragraph, questions_per_paragraph)
            if not isinstance(generated, list) or not all(
                isinstance(q, str) and q.strip() for q in generated
            ):
                raise ValueError("Question generator must return a list of nonempty strings")
            for question in generated:
                key = " ".join(question.casefold().split())
                if key not in seen:
                    seen.add(key)
                    questions.append(question.strip())
        if len(questions) < 3:
            raise ValueError(
                f"Leaf {leaf.label!r} needs at least 3 distinct questions; "
                "supply text/questions or clean the ToC"
            )
        rng.shuffle(questions)
        train_count = min(max(1, int(len(questions) * train_fraction)), len(questions) - 2)
        dev_count = min(
            max(1, int(len(questions) * dev_fraction)), len(questions) - train_count - 1
        )
        for i, query in enumerate(questions):
            split = "train" if i < train_count else "dev" if i < train_count + dev_count else "test"
            rows.append(
                {"query": query, "split": split, "document_id": document.id, "section_id": leaf.id}
            )
    if not rows:
        raise ValueError("No leaf paragraphs were found")
    return rows


class LocalQuestionGenerator:
    """Local Mixtral by default; any compatible causal instruction model can be supplied."""

    def __init__(self, model_name: str = PAPER_QA_MODEL, device: str = "auto", seed: int = 42):
        from transformers import set_seed

        from .neural import _dependencies, choose_device, model_dtype

        self.torch, auto_model, auto_tokenizer = _dependencies()
        set_seed(seed)
        self.device = choose_device(self.torch, device)
        self.tokenizer = auto_tokenizer.from_pretrained(model_name, trust_remote_code=False)
        self.model = (
            auto_model.from_pretrained(
                model_name,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=model_dtype(self.torch, "auto", self.device),
            )
            .to(self.device)
            .eval()
        )

    def __call__(self, paragraph: str, count: int) -> list[str]:
        prompt = (
            f"Write {count} diverse questions answerable solely from this paragraph. "
            "Cover the important information. Return a JSON array of question strings, "
            "without answers, headings, or commentary.\nParagraph:\n" + paragraph
        )
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        length = inputs["input_ids"].shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if length + 1024 > limit:
            raise ValueError("Paragraph exceeds QA model context; clean/split the paragraph first")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **{k: v.to(self.device) for k, v in inputs.items()},
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(generated[0, length:], skip_special_tokens=True).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            questions = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "QA model did not return a JSON array; choose a stronger model/retry"
            ) from error
        if not isinstance(questions, list):
            raise ValueError("QA model must return a JSON array")
        return questions
