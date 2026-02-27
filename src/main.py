"""
main.py - SubJav CLI 入口
"""
import gzip
import json
import tomllib
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
    hybrid = "hybrid"


def _load_config() -> dict:
    """從專案根目錄讀取 subjav.toml，回傳 [defaults] 區段（無檔案則回傳空 dict）。"""
    path = Path(__file__).parent.parent / "subjav.toml"
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f).get("defaults", {})
    return {}


def _find_videos(directory: Path) -> list[Path]:
    """遞迴搜尋目錄內所有影片檔，依路徑排序。"""
    from .transcribe import VIDEO_EXTENSIONS
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def _process_video(
    video: Path,
    cfg: dict,
    backend: Backend,
    hard_sub: bool,
    srt_only: bool,
    force: bool,
    output_dir: Path | None,
) -> None:
    """轉錄單一影片 → 翻譯 → 輸出 SRT（可選嵌入字幕）。"""
    from .transcribe import transcribe, Segment
    from .translate import translate
    from .subtitle import build_srt, save_srt
    from .embed import embed_soft, embed_hard

    if not video.exists():
        typer.echo(f"找不到檔案: {video}", err=True)
        return

    typer.echo(f"\n{'='*60}\n處理: {video.name}\n{'='*60}")

    if output_dir is not None:
        work_dir = output_dir / video.stem
    elif cfg.get("output_beside_video", False):
        work_dir = video.parent
    else:
        work_dir = Path("output") / video.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = video.stem

    def _is_hallucination(text: str) -> bool:
        if len(text) < 20:
            return False
        ratio = len(text) / len(gzip.compress(text.encode()))
        return ratio > 2.4

    def _load_or_transcribe(segments_path: Path, label: str, transcribe_fn, audio_path: Path) -> list:
        if segments_path.exists():
            typer.echo(f"已有轉錄結果，跳過 {label} 轉錄: {segments_path.name}")
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

    audio_path        = work_dir / f"{stem}.wav"
    whisper_segs_path = work_dir / f"{stem}_whisper_segments.json"
    whisper_srt_path  = work_dir / f"{stem}_whisper.srt"
    qwen3_segs_path   = work_dir / f"{stem}_qwen3_segments.json"
    qwen3_srt_path    = work_dir / f"{stem}_qwen3.srt"
    hybrid_srt_path   = work_dir / f"{stem}_hybrid.srt"

    run_whisper = backend in (Backend.whisper, Backend.hybrid)
    run_qwen3 = backend in (Backend.qwen3, Backend.hybrid)

    # ── 清除快取（--force）─────────────────────────────────────────────────
    if force:
        targets = [audio_path]
        if run_whisper:
            targets += [whisper_segs_path, whisper_srt_path]
        if run_qwen3:
            targets += [qwen3_segs_path, qwen3_srt_path]
        if backend == Backend.hybrid:
            targets += [hybrid_srt_path]
        for p in targets:
            if p.exists():
                p.unlink()
                typer.echo(f"刪除: {p.name}")

    # ── 跑各路 ASR ─────────────────────────────────────────────────────────
    if backend == Backend.whisper:
        _run_backend("Whisper", transcribe, whisper_segs_path, whisper_srt_path)

    if backend == Backend.qwen3:
        from .transcribe_qwen3 import transcribe_qwen3
        _run_backend("Qwen3-ASR", transcribe_qwen3, qwen3_segs_path, qwen3_srt_path)

    if backend == Backend.hybrid:
        if hybrid_srt_path.exists():
            typer.echo(f"Hybrid SRT 已存在，跳過: {hybrid_srt_path.name}")
        else:
            from .transcribe_qwen3 import transcribe_qwen3 as _tq3
            from .translate import merge_with_llm
            whisper_segs = _load_or_transcribe(
                whisper_segs_path, "Whisper", transcribe, audio_path
            )
            qwen3_segs = _load_or_transcribe(
                qwen3_segs_path, "Qwen3-ASR", _tq3, audio_path,
            )
            if not whisper_segs:
                typer.echo("[Hybrid] Whisper 無轉錄結果，無法執行合併", err=True)
            elif not qwen3_segs:
                typer.echo("[Hybrid] Qwen3 無轉錄結果，無法執行合併", err=True)
            else:
                typer.echo("[Hybrid] LLM 合併中（Whisper 時間戳 + Qwen3 文字）...")
                merged = merge_with_llm(whisper_segs, qwen3_segs)
                filtered, translations = _filter_and_translate(merged, "Hybrid")
                srt_content = build_srt(filtered, translations)
                save_srt(srt_content, hybrid_srt_path)
                typer.echo(f"Hybrid SRT: {hybrid_srt_path}")

    if srt_only:
        return

    # ── 嵌入字幕 ──────────────────────────────────────────────────────────
    if backend == Backend.hybrid:
        srt_path = hybrid_srt_path
    elif run_whisper:
        srt_path = whisper_srt_path
    else:
        srt_path = qwen3_srt_path
    if not srt_path.exists():
        typer.echo(f"SRT 不存在，無法嵌入: {srt_path.name}", err=True)
        return

    typer.echo("嵌入字幕...")
    out_video = work_dir / f"{stem}_subtitled.mp4"
    if hard_sub:
        embed_hard(video, srt_path, out_video)
    else:
        embed_soft(video, srt_path, out_video)
    typer.echo(f"影片已輸出: {out_video}")


@app.command()
def process(
    target: str | None = typer.Argument(None, help="影片路徑（省略時自動掃描 video_dir）"),
    hard_sub: bool = typer.Option(False, "--hard-sub", help="嵌入硬字幕（燒錄進畫面）"),
    srt_only: bool = typer.Option(False, "--srt-only", help="只輸出 SRT，不嵌入影片"),
    force: bool = typer.Option(False, "--force", "-f", help="強制重新轉錄，刪除已有的 segments/srt"),
    backend: Backend | None = typer.Option(None, "--backend", "-b", help="ASR 後端：whisper / qwen3 / hybrid"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="輸出目錄（預設：影片所在目錄）"),
) -> None:
    """轉錄日語 → 翻譯中文 → 輸出 SRT（可選嵌入字幕）

    ASR 後端（--backend）：
      whisper  Whisper Large V3，時間戳準確（預設）
      qwen3    Qwen3-ASR，日語辨識品質較好
      hybrid   Whisper 時間戳 + Qwen3 文字品質，推薦
    """
    cfg = _load_config()

    # CLI 參數優先；未指定則從 config 讀；最後才用 hardcoded 預設值
    if backend is None:
        backend = Backend(cfg.get("backend", "whisper"))
    srt_only = srt_only or cfg.get("srt_only", False)
    hard_sub = hard_sub or cfg.get("hard_sub", False)

    # 決定要處理哪些影片
    if target is not None:
        video = Path(target)
        if not video.is_absolute() and "video_dir" in cfg:
            video = Path(cfg["video_dir"]) / video
        videos = [video]
    else:
        video_dir_str = cfg.get("video_dir")
        if not video_dir_str:
            typer.echo("請指定影片路徑，或在 subjav.toml 設定 video_dir", err=True)
            raise typer.Exit(1)
        videos = _find_videos(Path(video_dir_str))
        if not videos:
            typer.echo(f"在 {video_dir_str} 找不到影片檔", err=True)
            raise typer.Exit(1)
        typer.echo(f"找到 {len(videos)} 部影片")

    for video in videos:
        _process_video(video, cfg, backend, hard_sub, srt_only, force, output_dir)


if __name__ == "__main__":
    app()
