"""
transcribe_qwen3.py - 使用 mlx-audio (Qwen3-ASR-1.7B-8bit + ForcedAligner) 轉錄日語音軌
回傳與 transcribe.py 相同的 list[Segment] 格式

流程：
  1. 抽音軌（共用快取）
  2. 切 120s chunk
  3. ASR model → 每個 chunk 的文字（含標點）
  4. ForcedAligner → 詞級時間戳（標點已被過濾，items 只有字/詞）
  5. 按原始文字的 。！？ 切句，對應 items index → Segment
"""
import re
from pathlib import Path

import numpy as np

from .transcribe import Segment, prepare_audio

QWEN3_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN3_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

_SENT_END_RE = re.compile(r"([。！？])")
_MIN_SEGMENT_SEC = 0.5


def _split_sentences(text: str) -> list[str]:
    """按 。！？ 切句，標點保留在句尾。"""
    parts = _SENT_END_RE.split(text)
    sentences: list[str] = []
    cur = ""
    for part in parts:
        cur += part
        if part in "。！？":
            s = cur.strip()
            if s:
                sentences.append(s)
            cur = ""
    if cur.strip():
        sentences.append(cur.strip())
    return sentences


def _count_clean_tokens(text: str) -> int:
    """
    統計 nagisa 對 text 產生的 clean token 數。
    與 ForcedAligner 內部 tokenize_japanese + clean_token 邏輯一致。
    """
    import unicodedata
    import nagisa

    def is_kept(ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    words = nagisa.tagging(text).words
    return sum(1 for w in words if any(is_kept(ch) for ch in w))


def _items_to_segments(items, text: str, chunk_offset: float) -> list[Segment]:
    """
    把 ForcedAligner 的詞級 items（無標點）＋原始文字（含標點），
    切成句子級 Segment，加上 chunk offset。

    做法：
      1. 把 text 按 。！？ 切句
      2. 每句用 nagisa 數 clean token 數量 n
      3. 從 items 取 n 個，得到該句的時間範圍
    """
    sentences = _split_sentences(text)
    if not sentences or not items:
        return []

    segments: list[Segment] = []
    item_idx = 0

    for sentence in sentences:
        n = _count_clean_tokens(sentence)
        if n == 0 or item_idx >= len(items):
            continue

        sent_items = items[item_idx: item_idx + n]
        item_idx += n

        start = sent_items[0].start_time
        end = sent_items[-1].end_time
        dur = end - start

        seg = Segment(
            start=round(chunk_offset + start, 3),
            end=round(chunk_offset + end, 3),
            text=sentence,
        )
        if dur < _MIN_SEGMENT_SEC and segments:
            prev = segments[-1]
            segments[-1] = Segment(start=prev.start, end=seg.end, text=prev.text + sentence)
        else:
            segments.append(seg)

    return segments


def transcribe_qwen3(input_path: str | Path, audio_save_path: Path | None = None) -> list[Segment]:
    """
    使用 Qwen3-ASR + ForcedAligner 轉錄影片或音訊，回傳 list[Segment]。

    Args:
        input_path: 影片或音訊檔路徑
        audio_save_path: 音檔儲存路徑（None 則用暫存檔）

    Returns:
        list[Segment]，每個 Segment 含 start（秒）、end（秒）、text（日語）
    """
    import mlx.core as mx
    from mlx_audio.stt.utils import load_audio, load_model as stt_load
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到檔案: {input_path}")

    with prepare_audio(input_path, audio_save_path, "qwen3-asr") as audio_path:
        # 2. 載入音訊並切 chunk
        print("[qwen3-asr] 載入音訊...")
        audio_np = np.array(load_audio(str(audio_path)))
        chunks = split_audio_into_chunks(audio_np, sr=16000, chunk_duration=120.0)
        print(f"[qwen3-asr] 分為 {len(chunks)} 個 chunk")

        # 3. ASR：每個 chunk 取得文字
        print(f"[qwen3-asr] 載入 ASR 模型 ({QWEN3_ASR_MODEL})...")
        asr_model = stt_load(QWEN3_ASR_MODEL)

        chunk_data: list[tuple[np.ndarray, float, str]] = []
        for i, (chunk_audio, offset_sec) in enumerate(chunks):
            print(f"[qwen3-asr] ASR chunk {i+1}/{len(chunks)} (offset={offset_sec:.1f}s)...")
            result = asr_model.generate(
                chunk_audio,
                language="Japanese",
                max_tokens=8192,
                temperature=0.0,
                repetition_penalty=1.1,
                repetition_context_size=100,
                verbose=False,
            )
            chunk_data.append((chunk_audio, offset_sec, result.text.strip()))

        del asr_model
        mx.clear_cache()

        # 3.5 補標點：讓 ForcedAligner 能按句子對齊
        from .translate import punctuate_japanese
        print(f"[qwen3-asr] 補標點（共 {len(chunk_data)} 個 chunk）...")
        chunk_data = [
            (audio, offset, punctuate_japanese(text))
            for audio, offset, text in chunk_data
        ]

        # 4. ForcedAligner：詞級時間戳 → 句子級 Segment
        print(f"[qwen3-asr] 載入 ForcedAligner ({QWEN3_ALIGNER_MODEL})...")
        aligner = stt_load(QWEN3_ALIGNER_MODEL)

        all_segments: list[Segment] = []
        for i, (chunk_audio, offset_sec, text) in enumerate(chunk_data):
            if not text:
                print(f"[qwen3-asr] Chunk {i+1} 無文字，跳過")
                continue
            print(f"[qwen3-asr] Align chunk {i+1}/{len(chunk_data)} (offset={offset_sec:.1f}s)...")
            align_result = aligner.generate(chunk_audio, text=text, language="Japanese")
            segs = _items_to_segments(align_result.items, text, chunk_offset=offset_sec)
            all_segments.extend(segs)

        del aligner
        mx.clear_cache()

        print(f"[qwen3-asr] 完成，共 {len(all_segments)} 段")
        return all_segments
