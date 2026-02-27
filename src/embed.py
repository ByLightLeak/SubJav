"""
embed.py - 使用 ffmpeg 將字幕嵌入影片
"""
import subprocess
from pathlib import Path


HARD_SUB_FONT = "Noto Sans CJK TC"
HARD_SUB_FONT_SIZE = 20


def embed_soft(video_path: str | Path, srt_path: str | Path, output_path: str | Path) -> Path:
    """
    嵌入軟字幕（可切換，不燒錄進畫面）。

    Args:
        video_path: 原始影片路徑
        srt_path: SRT 字幕檔路徑
        output_path: 輸出影片路徑

    Returns:
        輸出影片 Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-c", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=chi",
        str(output_path),
    ]
    print(f"[embed] 嵌入軟字幕: {output_path.name}")
    _run_ffmpeg(cmd)
    return output_path


def embed_hard(video_path: str | Path, srt_path: str | Path, output_path: str | Path) -> Path:
    """
    嵌入硬字幕（燒錄進畫面）。

    Args:
        video_path: 原始影片路徑
        srt_path: SRT 字幕檔路徑
        output_path: 輸出影片路徑

    Returns:
        輸出影片 Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 路徑中的冒號需要跳脫（Windows 相容處理）
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

    vf = (
        f"subtitles={srt_escaped}:"
        f"force_style='FontName={HARD_SUB_FONT},FontSize={HARD_SUB_FONT_SIZE}'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        str(output_path),
    ]
    print(f"[embed] 嵌入硬字幕: {output_path.name}")
    _run_ffmpeg(cmd)
    return output_path


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 執行失敗:\n{result.stderr[-2000:]}")
    print(f"[embed] 完成")
