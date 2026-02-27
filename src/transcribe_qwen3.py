"""
transcribe_qwen3.py - 使用 mlx-audio (Qwen3-ASR-1.7B-8bit + ForcedAligner) 轉錄日語音軌
回傳與 transcribe.py 相同的 list[Segment] 格式

流程：
  1. 抽音軌（共用快取）
  2. 切 60s chunk
  3. ASR model → 每個 chunk 的文字（含標點）
  4. ForcedAligner → 詞級時間戳（標點已被過濾，items 只有字/詞）
  5. 按原始文字的 。！？ 切句，對應 items index → Segment
"""
import re
from pathlib import Path

import numpy as np

from .transcribe import Segment, prepare_audio, _filter_hallucinations

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


def _items_to_segments(items, text: str, chunk_offset: float) -> list[Segment]:
    """
    把 ForcedAligner 的詞級 items 按句子切成 Segment。

    直接用 item.text 的字元數累加來找句子邊界，不需要重跑 nagisa。
    ForcedAligner 的每個 item.text 已是 clean token（無標點），
    其字元數總和等於句子去標點後的字元數。
    """
    sentences = _split_sentences(text)
    if not sentences or not items:
        return []

    import unicodedata

    def _clean_len(s: str) -> int:
        """句子去掉標點後的字元數，對應 item.text 字元數之和。"""
        return sum(
            1 for ch in s
            if ch == "'" or unicodedata.category(ch)[:1] in ("L", "N")
        )

    segments: list[Segment] = []
    item_idx = 0

    for sentence in sentences:
        if item_idx >= len(items):
            break

        target_len = _clean_len(sentence)
        if target_len == 0:
            continue

        # 累加 item.text 字元數直到覆蓋整句
        accumulated = 0
        n = 0
        for item in items[item_idx:]:
            accumulated += len(item.text)
            n += 1
            if accumulated >= target_len:
                break

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
        chunks = split_audio_into_chunks(audio_np, sr=16000, chunk_duration=60.0)
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
                max_tokens=512,
                temperature=0.0,
                repetition_penalty=1.2,
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

        filtered = _filter_hallucinations(all_segments)
        print(f"[qwen3-asr] 完成，共 {len(all_segments)} 段（過濾後 {len(filtered)} 段）")
        return filtered
