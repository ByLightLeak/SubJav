"""
main.py - SubJav CLI 入口
"""
from enum import Enum
from pathlib import Path

import typer

app = typer.Typer(
    name="subjav",
    help="本地日語影片字幕自動生成工具（ASR + LLM 翻譯）",
    add_completion=False,
)


class Backend(str, Enum):
    whisper = "whisper"
    qwen3 = "qwen3"
    both = "both"
    hybrid = "hybrid"


@app.command()
def process(
    target: str = typer.Argument(..., help="本地影片路徑"),
    hard_sub: bool = typer.Option(False, "--hard-sub", help="嵌入硬字幕（燒錄進畫面）"),
    srt_only: bool = typer.Option(False, "--srt-only", help="只輸出 SRT，不嵌入影片"),
    force: bool = typer.Option(False, "--force", "-f", help="強制重新轉錄，刪除已有的 segments/srt"),
    backend: Backend = typer.Option(Backend.whisper, "--backend", "-b", help="ASR 後端：whisper / qwen3 / both / hybrid"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="輸出目錄"),
) -> None:
    """轉錄日語 → 翻譯中文 → 輸出 SRT（可選嵌入字幕）

    ASR 後端（--backend）：
      whisper  Whisper Large V3，時間戳準確（預設）
      qwen3    Qwen3-ASR，日語辨識品質較好
      both     兩者都跑，各輸出一份 SRT 供比較
      hybrid   Whisper 時間戳 + Qwen3 文字品質，推薦
    """
    from .transcribe import transcribe
    from .translate import translate
    from .subtitle import build_srt, save_srt
    from .embed import embed_soft, embed_hard

    video = Path(target)
    if not video.exists():
        typer.echo(f"找不到檔案: {video}", err=True)
        raise typer.Exit(1)
    work_dir = output_dir / video.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = video.stem

    import json
    import gzip as _gzip

    def _is_hallucination(text: str) -> bool:
        if len(text) < 20:
            return False
        ratio = len(text) / len(_gzip.compress(text.encode()))
        return ratio > 2.4

    def _load_or_transcribe(segments_path: Path, label: str, transcribe_fn, audio_path: Path) -> list:
        if segments_path.exists():
            typer.echo(f"已有轉錄結果，跳過 {label} 轉錄: {segments_path.name}")
            from .transcribe import Segment
            data = json.loads(segments_path.read_text(encoding="utf-8"))
            return [Segment(**d) for d in data]
        typer.echo(f"[{label}] 轉錄: {video.name}")
        segs = transcribe_fn(video, audio_save_path=audio_path)
        if not segs:
            typer.echo(f"[{label}] 未偵測到任何字幕段落", err=True)
            return []
        segments_path.write_text(
            json.dumps([{"start": s.start, "end": s.end, "text": s.text} for s in segs],
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"[{label}] 轉錄結果: {segments_path.name}")
        return segs

    def _filter_and_translate(segs: list, label: str) -> tuple:
        filtered = [s for s in segs if not _is_hallucination(s.text)]
        dropped = len(segs) - len(filtered)
        if dropped:
            typer.echo(f"[{label}] 過濾幻覺 segments：{dropped} 段")
        typer.echo(f"[{label}] 翻譯 {len(filtered)} 段字幕...")
        translations = translate([s.text for s in filtered])
        return filtered, translations

    def _run_backend(label: str, transcribe_fn, segs_path: Path, srt_path: Path) -> None:
        if srt_path.exists():
            typer.echo(f"{label} SRT 已存在，跳過: {srt_path.name}")
            return
        segs = _load_or_transcribe(segs_path, label, transcribe_fn, audio_path)
        if not segs:
            typer.echo(f"{label} 無結果，跳過", err=True)
            return
        filtered, translations = _filter_and_translate(segs, label)
        srt_content = build_srt(filtered, translations)
        save_srt(srt_content, srt_path)
        typer.echo(f"{label} SRT: {srt_path}")

    audio_path = work_dir / f"{stem}.wav"

    run_whisper = backend in (Backend.whisper, Backend.both, Backend.hybrid)
    run_qwen3 = backend in (Backend.qwen3, Backend.both, Backend.hybrid)

    # ── 清除快取（--force）─────────────────────────────────────────────────
    if force:
        targets = [audio_path]
        if run_whisper:
            targets += [work_dir / f"{stem}_whisper_segments.json",
                        work_dir / f"{stem}_whisper.srt"]
        if run_qwen3:
            targets += [work_dir / f"{stem}_qwen3_segments.json",
                        work_dir / f"{stem}_qwen3.srt"]
        if backend == Backend.hybrid:
            targets += [work_dir / f"{stem}_hybrid.srt"]
        for p in targets:
            if p.exists():
                p.unlink()
                typer.echo(f"刪除: {p.name}")

    # ── 跑各路 ASR ─────────────────────────────────────────────────────────
    if run_whisper and backend != Backend.hybrid:
        _run_backend(
            "Whisper",
            transcribe,
            work_dir / f"{stem}_whisper_segments.json",
            work_dir / f"{stem}_whisper.srt",
        )

    if run_qwen3 and backend != Backend.hybrid:
        from .transcribe_qwen3 import transcribe_qwen3
        _run_backend(
            "Qwen3-ASR",
            transcribe_qwen3,
            work_dir / f"{stem}_qwen3_segments.json",
            work_dir / f"{stem}_qwen3.srt",
        )

    if backend == Backend.hybrid:
        hybrid_srt_path = work_dir / f"{stem}_hybrid.srt"
        if hybrid_srt_path.exists():
            typer.echo(f"Hybrid SRT 已存在，跳過: {hybrid_srt_path.name}")
        else:
            from .transcribe_qwen3 import transcribe_qwen3 as _tq3
            from .translate import merge_with_llm
            whisper_segs = _load_or_transcribe(
                work_dir / f"{stem}_whisper_segments.json", "Whisper", transcribe, audio_path
            )
            qwen3_segs = _load_or_transcribe(
                work_dir / f"{stem}_qwen3_segments.json", "Qwen3-ASR",
                _tq3, audio_path,
            )
            if whisper_segs and qwen3_segs:
                typer.echo("[Hybrid] LLM 合併中（Whisper 時間戳 + Qwen3 文字）...")
                merged = merge_with_llm(whisper_segs, qwen3_segs)
                filtered, translations = _filter_and_translate(merged, "Hybrid")
                srt_content = build_srt(filtered, translations)
                save_srt(srt_content, hybrid_srt_path)
                typer.echo(f"Hybrid SRT: {hybrid_srt_path}")

    if srt_only:
        return

    # ── 嵌入字幕（單一路才嵌入）───────────────────────────────────────────
    if backend == Backend.both:
        typer.echo("both 模式不嵌入影片，請自行選擇 SRT 後重跑")
        return

    if backend == Backend.hybrid:
        srt_path = work_dir / f"{stem}_hybrid.srt"
    elif run_whisper:
        srt_path = work_dir / f"{stem}_whisper.srt"
    else:
        srt_path = work_dir / f"{stem}_qwen3.srt"
    if not srt_path.exists():
        typer.echo(f"SRT 不存在，無法嵌入: {srt_path.name}", err=True)
        raise typer.Exit(1)

    typer.echo("嵌入字幕...")
    out_video = work_dir / f"{stem}_subtitled.mp4"
    if hard_sub:
        embed_hard(video, srt_path, out_video)
    else:
        embed_soft(video, srt_path, out_video)
    typer.echo(f"影片已輸出: {out_video}")


if __name__ == "__main__":
    app()
