"""
transcribe.py - 使用 mlx-audio (Whisper Large V3) 轉錄日語音軌
"""
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Segment:
    start: float   # 秒
    end: float     # 秒
    text: str      # 原文（日語）
    translated: str = field(default="")  # 翻譯後（中文）


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv"}
MLX_MODEL = "mlx-community/whisper-large-v3-mlx"

# Whisper initial prompt：引導模型輸出日語，抑制幻覺與英文混入
INITIAL_PROMPT = "日本語のアダルトビデオ。あっ、はぁ、んっ、うぁ、ふぁ、ひゃ、んー。ダメ、イク、イキそう、気持ちいい、中に出さないで。射精、中出し、フェラ、クンニ、おちんちん、乳首、性感帯、クリトリス。"


def _ensure_whisper_processor() -> None:
    """
    mlx-community/whisper-large-v3-mlx 是給舊版 mlx-whisper 套件打包的，
    缺少 mlx-audio 需要的 WhisperProcessor 檔案。
    首次載入失敗時自動從 openai/whisper-large-v3 下載補齊（小檔，約 5MB）。
    """
    from huggingface_hub import snapshot_download
    from transformers import WhisperProcessor

    snapshot_path = Path(snapshot_download(MLX_MODEL, local_files_only=True))
    if (snapshot_path / "preprocessor_config.json").exists() or \
       (snapshot_path / "processor_config.json").exists():
        return

    print("[transcribe] 自動補充 WhisperProcessor 檔案（首次執行）...")
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    proc.save_pretrained(str(snapshot_path))
    print("[transcribe] WhisperProcessor 補充完成")


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


def transcribe(input_path: str | Path, audio_save_path: Path | None = None) -> List[Segment]:
    """
    轉錄影片或音訊檔案。

    Args:
        input_path: 影片或音訊檔路徑
        audio_save_path: 音檔儲存路徑（None 則用暫存檔，影片輸入才有效）

    Returns:
        List[Segment]，每個 Segment 含 start、end、text
    """
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model as stt_load

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到檔案: {input_path}")

    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS

    # 1. 抽音軌
    if is_video:
        if audio_save_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = Path(tmp.name)
            tmp.close()
            cleanup = True
        else:
            audio_path = audio_save_path
            cleanup = False
        if audio_path.exists():
            print(f"[transcribe] 音檔已存在，跳過抽取: {audio_path.name}")
        else:
            try:
                extract_audio(input_path, audio_path)
            except Exception:
                if cleanup:
                    audio_path.unlink(missing_ok=True)
                raise
    else:
        audio_path = input_path
        cleanup = False

    try:
        print(f"[transcribe] 開始轉錄（mlx-audio Whisper Large V3）...")
        try:
            model = stt_load(MLX_MODEL)
        except ValueError as e:
            if "Processor not found" not in str(e):
                raise
            _ensure_whisper_processor()
            model = stt_load(MLX_MODEL)

        result = model.generate(
            str(audio_path),
            language="ja",
            initial_prompt=INITIAL_PROMPT,
            verbose=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.4,
            hallucination_silence_threshold=2.0,
            condition_on_previous_text=True,
            carry_initial_prompt=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            prepend_punctuations="\"'¿([{-「『【〔（",
            append_punctuations="\"'.。,，!！?？:：\")]}、」』】〕）～…",
        )

        # 釋放模型
        del model
        mx.clear_cache()

        all_segments: List[Segment] = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in (result.segments or [])
            if s["text"].strip()
        ]
        print(f"[transcribe] 完成，共 {len(all_segments)} 段")
        return all_segments
    finally:
        if cleanup:
            audio_path.unlink(missing_ok=True)
