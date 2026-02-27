"""
transcribe.py - 使用 mlx-whisper (Whisper Large V3) 轉錄日語音軌
"""
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float   # 秒
    end: float     # 秒
    text: str      # 原文（日語）


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}
MLX_MODEL = "mlx-community/whisper-large-v3-mlx"

# Whisper initial prompt：引導模型輸出日語，抑制幻覺與英文混入
INITIAL_PROMPT = "日本語のアダルトビデオ。あっ、はぁ、んっ、うぁ、ふぁ、ひゃ、うん、あーん、おっ、いやん、ふぅ、ひぃ、んぁ、あぁ。ダメ。気持ちいい。もっと奥まで。イキそう。中に出さないで。感じてる。濡れてる。イっちゃう。舐めていい？乳首が好き。恥ずかしい。出していい？気持ちよかった。フェラ、クンニ、手マン、手コキ、パイズリ、アナル、騎乗位、バック、中出し、顔射、潮吹き、射精。クリトリス、おまんこ、チンポ、おちんちん、膣、乳頭、おっぱい、亀頭、お尻、挿入、性感帯。"


def _filter_hallucinations(segments: list[Segment]) -> list[Segment]:
    """
    移除 Whisper 幻覺片段：
    1. 反轉時間戳（end <= start）→ 必然是幻覺
    2. 連續重複文字（相同文字連續出現 N 次）→ Whisper loop 幻覺
    """
    # 1. 移除反轉時間戳
    result = [s for s in segments if s.end > s.start]
    inverted = len(segments) - len(result)
    if inverted:
        print(f"[transcribe] 過濾反轉時間戳: {inverted} 段")

    # 2. 移除連續重複（同一文字連續出現超過 2 次則從第 3 次起刪除）
    deduped: list[Segment] = []
    consecutive = 0
    for seg in result:
        if deduped and seg.text == deduped[-1].text:
            consecutive += 1
            if consecutive >= 2:
                continue  # 同一句連出現 3 次以上才刪
        else:
            consecutive = 0
        deduped.append(seg)

    repeated = len(result) - len(deduped)
    if repeated:
        print(f"[transcribe] 過濾連續重複幻覺: {repeated} 段")

    return deduped


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """用 ffmpeg 從影片抽取音軌為 wav，存到指定路徑"""
    print(f"[transcribe] 抽取音軌 → {audio_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 音軌抽取失敗:\n{result.stderr}")
    print(f"[transcribe] 音檔已儲存: {audio_path}")


@contextmanager
def prepare_audio(input_path: Path, audio_save_path: Path | None, label: str):
    """影片 → WAV（共用快取）；非影片直接 yield；暫存檔使用後自動清除。"""
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        yield input_path
        return

    if audio_save_path is not None:
        audio_path = audio_save_path
        cleanup = False
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = Path(tmp.name)
        tmp.close()
        cleanup = True

    if audio_path.exists():
        print(f"[{label}] 音檔已存在，跳過抽取: {audio_path.name}")
    else:
        try:
            extract_audio(input_path, audio_path)
        except Exception:
            if cleanup:
                audio_path.unlink(missing_ok=True)
            raise

    try:
        yield audio_path
    finally:
        if cleanup:
            audio_path.unlink(missing_ok=True)


def transcribe(input_path: str | Path, audio_save_path: Path | None = None) -> list[Segment]:
    """
    轉錄影片或音訊檔案。

    Args:
        input_path: 影片或音訊檔路徑
        audio_save_path: 音檔儲存路徑（None 則用暫存檔，影片輸入才有效）

    Returns:
        list[Segment]，每個 Segment 含 start、end、text
    """
    import mlx.core as mx
    import mlx_whisper

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到檔案: {input_path}")

    # word_timestamps 會造成記憶體持續成長，限制 Metal cache
    mx.set_cache_limit(100_000_000)

    with prepare_audio(input_path, audio_save_path, "transcribe") as audio_path:
        print("[transcribe] 開始轉錄（mlx-whisper Whisper Large V3）...")

        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=MLX_MODEL,
            language="ja",
            initial_prompt=INITIAL_PROMPT,
            verbose=False,
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.4,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            prepend_punctuations="\"'¿([{-「『【〔（",
            append_punctuations="\"'.。,，!！?？:：\")]}、」』】〕）～…",
        )

        mx.clear_cache()

        all_segments: list[Segment] = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in (result.get("segments") or [])
            if s["text"].strip()
        ]
        filtered = _filter_hallucinations(all_segments)
        print(f"[transcribe] 完成，共 {len(all_segments)} 段（過濾後 {len(filtered)} 段）")
        return filtered
