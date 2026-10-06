"""Speaker embeddings from the labeller itself: one vector per decoded turn.

Long-form decoding labels speakers per chunk (`<SPK_1>` means "the first voice
in this chunk"), so chunks must be linked into recording-level speakers. This
module gives every turn a vector that stays put for one person across chunks:

    decoder hidden states over a turn's tokens  ->  mean  ->  SpeakerHead  ->  unit vector

trained with a within-meeting supervised contrastive loss (turns of one
speaker pull together, other speakers OF THE SAME MEETING push apart -- linking
only ever separates the 3-5 people of one recording) and, optionally, a
distillation loss toward ECAPA-TDNN embeddings of the turn's clean headset
audio, so the head starts from ECAPA's open-set voice identity instead of
AMI's ~150 training speakers. Linking then clusters these vectors.

The pieces are pure functions over tensors so they can be tested without a model:
turn_speakers (which meeting speaker each target turn is), turn_metadata (each
turn's token span in a batch), supcon_loss, MeetingBatchSampler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

ECAPA_DIM = 192


@dataclass(frozen=True)
class EmbeddingConfig:
    """Speaker-embedding head (configs/speaker_asr `embedding:`)."""

    # Output size of the speaker vector.
    dim: int = 256
    # Weight of the contrastive loss next to the transcription loss.
    weight: float = 0.1
    # SupCon temperature.
    temperature: float = 0.1
    # Weight of the ECAPA distillation loss (0 = off).
    distill_weight: float = 0.0
    # Meetings per training batch: the rest of the batch is more windows of them.
    meetings_per_batch: int = 2
    # Eval windows scored for eval_embed/auc (same-meeting turn pairs); 0 = none.
    eval_rows: int = 200


class SpeakerHead(nn.Module):
    """Mean decoder state of a turn -> unit speaker vector (+ an ECAPA-space projection)."""

    def __init__(self, hidden_size: int, dim: int = 256, distill: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, dim)
        )
        self.to_ecapa = nn.Linear(dim, ECAPA_DIM) if distill else None

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.proj(pooled.float())
        teacher_space = self.to_ecapa(z) if self.to_ecapa is not None else None
        return F.normalize(z, dim=-1), teacher_space


def turn_speakers(parts: list[dict], texts: dict[str, str]) -> tuple[list, list[list[str]]]:
    """(speaker per target turn, utterance ids per turn), in the target's turn order.

    Mirrors data.format_target: parts ordered by (offset, dur), parts with no
    words dropped, consecutive parts of one speaker merged into one turn.
    """
    ordered = sorted(parts, key=lambda p: (p["offset_s"], p["dur_s"]))
    speakers: list = []
    ids: list[list[str]] = []
    for p in ordered:
        if not texts.get(p["id"], "").strip():
            continue
        if speakers and speakers[-1] == p["speaker"]:
            ids[-1].append(p["id"])
        else:
            speakers.append(p["speaker"])
            ids.append([p["id"]])
    return speakers, ids


def turn_spans(
    input_ids: torch.Tensor, labels: torch.Tensor, speaker_ids: set[int], end_ids: set[int]
) -> list[tuple[int, int]]:
    """[start, end) of each supervised turn: from its speaker token to the next one.

    Only the label region counts (a context prefix is masked, so its speaker
    tokens are ignored); trailing end tokens (<|im_end|>, ...) are excluded.
    """
    region = (labels != -100).nonzero().flatten().tolist()
    if not region:
        return []
    starts = [p for p in region if int(input_ids[p]) in speaker_ids]
    last = max((p for p in region if int(input_ids[p]) not in end_ids), default=region[-1])
    return [(s, (starts[k + 1] if k + 1 < len(starts) else last + 1)) for k, s in enumerate(starts)]


def turn_metadata(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    speaker_ids: set[int],
    end_ids: set[int],
    rows: list[dict],
) -> dict | None:
    """Batch-level turn bookkeeping for the embedding loss, or None if no row qualifies.

    rows[i] has `turn_speakers` and `group` (meeting). A row whose number of
    supervised speaker tokens differs from its turn count is skipped. Returns
    turn_index [B, T] (turn number at each token, -1 elsewhere), turn_speaker
    and turn_group [N] (int ids within the batch), and turn_ecapa [N, 192]
    when every kept row carries `turn_ecapa`.
    """
    turn_index = torch.full(input_ids.shape, -1, dtype=torch.long)
    speakers, groups, ecapa, keys = [], [], [], []
    speaker_codes: dict = {}
    group_codes: dict = {}
    n = 0
    for b, row in enumerate(rows):
        spk = row.get("turn_speakers")
        if not spk:
            continue
        spans = turn_spans(input_ids[b], labels[b], speaker_ids, end_ids)
        if len(spans) != len(spk):
            continue
        g = group_codes.setdefault(row.get("group"), len(group_codes))
        for k, (s, e) in enumerate(spans):
            turn_index[b, s:e] = n
            speakers.append(
                speaker_codes.setdefault((row.get("group"), spk[k]), len(speaker_codes))
            )
            groups.append(g)
            ecapa.append(row["turn_ecapa"][k] if row.get("turn_ecapa") is not None else None)
            keys.append((str(row.get("group")), str(spk[k]), b))
            n += 1
    if n == 0:
        return None
    out = {
        "turn_index": turn_index,
        "turn_speaker": torch.tensor(speakers, dtype=torch.long),
        "turn_group": torch.tensor(groups, dtype=torch.long),
        # (meeting, speaker, row in batch) as strings: batch-independent identity for eval
        "turn_keys": keys,
    }
    if all(e is not None for e in ecapa):
        out["turn_ecapa"] = torch.tensor(ecapa, dtype=torch.float32)
    return out


def pool_turns(hidden: torch.Tensor, turn_index: torch.Tensor, n_turns: int) -> torch.Tensor:
    """Mean hidden state per turn: hidden [B, T, H], turn_index [B, T] -> [N, H]."""
    flat_h = hidden.reshape(-1, hidden.shape[-1]).float()
    flat_i = turn_index.reshape(-1).to(hidden.device)
    keep = flat_i >= 0
    sums = torch.zeros(n_turns, flat_h.shape[-1], device=hidden.device).index_add_(
        0, flat_i[keep], flat_h[keep]
    )
    counts = torch.zeros(n_turns, device=hidden.device).index_add_(
        0, flat_i[keep], torch.ones(int(keep.sum()), device=hidden.device)
    )
    return sums / counts.clamp(min=1).unsqueeze(-1)


def supcon_loss(
    emb: torch.Tensor, speaker: torch.Tensor, group: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Supervised contrastive loss with negatives drawn only from the anchor's meeting.

    emb [N, D] unit vectors. For anchor i, positives are other turns of the same
    speaker; the softmax runs over every other turn of the same meeting. Anchors
    with no positive, or no negative, are left out. Returns 0 if none remain.
    """
    speaker, group = speaker.to(emb.device), group.to(emb.device)
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same_group = (group[:, None] == group[None, :]) & ~eye
    positive = (speaker[:, None] == speaker[None, :]) & ~eye
    negative = same_group & ~positive
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return emb.sum() * 0.0
    logits = sim.masked_fill(~same_group, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = (log_prob.masked_fill(~positive, 0.0)).sum(1) / positive.sum(1).clamp(min=1)
    return -pos_log_prob[valid].mean()


def distill_loss(teacher_space: torch.Tensor, ecapa: torch.Tensor) -> torch.Tensor:
    """1 - cosine between the head's ECAPA-space projection and the ECAPA target."""
    return (1.0 - F.cosine_similarity(teacher_space, ecapa.to(teacher_space.device), dim=-1)).mean()


# Keys the collator adds for the head; the trainer pops them before model(**inputs).
META_KEYS = ("turn_index", "turn_speaker", "turn_group", "turn_ecapa", "turn_keys")


class MeetingBatchSampler(torch.utils.data.Sampler):
    """Index order whose consecutive `batch_size` blocks hold `meetings` meetings each.

    The DataLoader cuts the stream into batches of batch_size, so every batch
    has several windows of the same few meetings -- the contrastive loss needs
    a speaker to recur within a batch. Every index appears once per epoch.
    """

    def __init__(self, groups: list, batch_size: int, meetings: int = 2, seed: int = 0):
        self.batch_size, self.meetings, self.seed = batch_size, max(1, meetings), seed
        self.by_group: dict = {}
        for i, g in enumerate(groups):
            self.by_group.setdefault(g, []).append(i)
        self.epoch = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_group.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {g: rng.sample(v, len(v)) for g, v in self.by_group.items()}
        per = max(1, self.batch_size // self.meetings)
        order = []
        while pools:
            chosen = rng.sample(sorted(pools, key=str), min(self.meetings, len(pools)))
            for g in chosen:
                order += pools[g][:per]
                pools[g] = pools[g][per:]
                if not pools[g]:
                    del pools[g]
        return iter(order)


def final_norm(model) -> nn.Module:
    """The decoder's final norm, whose output is the last hidden state the LM head reads."""
    for name, module in model.named_modules():
        if name.endswith("language_model.norm"):
            return module
    raise ValueError("no `language_model.norm` module found")


class LastHidden:
    """Forward hook that keeps the decoder's last hidden state of the latest call."""

    def __init__(self, model):
        self.value: torch.Tensor | None = None
        self.handle = final_norm(model).register_forward_hook(self._store)

    def _store(self, _module, _inputs, output):
        self.value = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self.handle.remove()


@torch.inference_mode()
def turn_embeddings(
    model, head: SpeakerHead, processor, audio, transcript: str, prefix: str = ""
) -> list[tuple[int, torch.Tensor]]:
    """[(label, unit vector)] for every turn of a decoded chunk, from one teacher-forced pass.

    `transcript` is the chunk's decoded `<SPK_n>...` text; `prefix` an optional
    context prefix ending in `<CONTINUE>` (as decoded with).
    """
    from scripts.speaker_asr.metrics import parse_turns
    from scripts.speaker_asr.model import n_speaker_tokens, speaker_token_ids
    from scripts.turn_aware.data import assistant_labels

    tok = processor.tokenizer
    spk_ids = speaker_token_ids(processor, n_speaker_tokens(processor))
    labels_by_id = {tid: i for i, tid in enumerate(spk_ids, 1)}
    conv = [[
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"language English<asr_text>{prefix}{transcript}"}]},
    ]]  # fmt: skip
    enc = processor.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt")
    labels = assistant_labels(
        enc["input_ids"], enc["attention_mask"], tok.convert_tokens_to_ids("<asr_text>"),
        tok.convert_tokens_to_ids("<|im_end|>"),
    )  # fmt: skip
    if prefix:  # mask the prefix through its <CONTINUE>, as the collator does
        cont = (enc["input_ids"][0] == tok.convert_tokens_to_ids("<CONTINUE>")).nonzero()
        if len(cont):
            labels[0, : int(cont[-1]) + 1] = -100
    end_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    spans = turn_spans(enc["input_ids"][0], labels[0], set(spk_ids), end_ids)
    if not spans or len(parse_turns(transcript)) == 0:
        return []
    hook = LastHidden(model)
    try:
        inputs = {
            k: (v.to(model.device, model.dtype) if v.is_floating_point() else v.to(model.device))
            for k, v in enc.items()
        }
        model(**inputs)
        hidden = hook.value
    finally:
        hook.remove()
    index = torch.full(enc["input_ids"].shape, -1, dtype=torch.long)
    for k, (s, e) in enumerate(spans):
        index[0, s:e] = k
    vecs, _ = head(pool_turns(hidden, index, len(spans)))
    return [
        (labels_by_id[int(enc["input_ids"][0, s])], vecs[k].cpu())
        for k, (s, _e) in enumerate(spans)
    ]
