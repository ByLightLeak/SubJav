"""
subtitle.py - 組合 SRT 格式字幕
"""
from pathlib import Path
from typing import List

from .transcribe import Segment


def _format_timestamp(seconds: float) -> str:
    """將秒數格式化為 SRT 時間戳 HH:MM:SS,mmm"""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: List[Segment], translations: List[str]) -> str:
    """
    組合 SRT 格式字串。

    Args:
        segments: 含時間戳的 Segment 列表
        translations: 對應的中文譯文列表

    Returns:
        標準 SRT 格式字串
    """
    if len(segments) != len(translations):
        raise ValueError(
            f"segments 數量 ({len(segments)}) 與 translations 數量 ({len(translations)}) 不符"
        )

    lines: List[str] = []
    for idx, (seg, trans) in enumerate(zip(segments, translations), start=1):
        start_ts = _format_timestamp(seg.start)
        end_ts = _format_timestamp(seg.end)
        text = trans if trans else seg.text  # 若翻譯為空則用原文
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # 空行分隔

    return "\n".join(lines)


def save_srt(content: str, output_path: str | Path) -> Path:
    """
    儲存 SRT 檔案。

    Args:
        content: SRT 格式字串
        output_path: 輸出路徑

    Returns:
        實際寫入的 Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"[subtitle] SRT 已儲存: {output_path}")
    return output_path
