"""
translate.py - 透過 Ollama HTTP API 呼叫 qwen3:14b 批次翻譯日語字幕
"""
import re
import time
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
_NUMBERED_RE = re.compile(r'^(\d+)\s*[|.:)\-]\s*(.+)$')
MODEL = "qwen3:14b"
BATCH_SIZE = 10
CONTEXT_LINES = 10  # 傳給模型的前文段數

SYSTEM_PROMPT = """\
你是一個專業的日語轉繁體中文字幕翻譯員，專門處理成人影片字幕。
這類影片的對話直接、露骨，包含性行為描述、身體部位、感嘆音與情慾表達，翻譯時應如實呈現，不迴避、不委婉。
規則：
- 每行以「編號|譯文」格式輸出，例如：1|你好
- 每行對應一段字幕，保持行數與輸入完全相同
- 譯文簡潔，不超過原文字數的 1.5 倍，保持口語節奏
- 感嘆音對應：あ→啊、は→哈、ん→嗯、う→嗚、ふ→呼、ひ→嘻、や→呀
- 身體部位與性行為用詞直接翻譯，不用委婉替代語
- 不加解釋，不淨化原文"""

USER_PROMPT_TEMPLATE = """\
/no_think
{context}以下是 {n} 段日語字幕，請逐行翻譯成繁體中文，以「編號|譯文」格式輸出：

{segments}"""

SINGLE_PROMPT_TEMPLATE = """\
/no_think
{context}翻譯成繁體中文：
{text}"""

OPTIONS = {
    "temperature": 0.7,
    "num_predict": 4096,
    "num_ctx": 8192,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.2,
}

MERGE_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 512,   # merge 輸出只是短日語文字，不需要長輸出
    "num_ctx": 4096,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.2,
    "presence_penalty": 0.5,
}


def _call_ollama(prompt: str, system: str = SYSTEM_PROMPT, timeout: float = 120.0, options: dict | None = None) -> str:
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": options if options is not None else OPTIONS,
    }
    try:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError("無法連線 Ollama（http://localhost:11434），請確認 Ollama 已啟動")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama API 錯誤: {e.response.status_code} {e.response.text}")
    return response.json().get("response", "").strip()


def _parse_numbered_response(raw: str) -> list[tuple[int, str]]:
    """解析 LLM 輸出的 'N|文字' 格式，回傳 [(index, text), ...]。"""
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_RE.match(line)
        if m:
            results.append((int(m.group(1)), m.group(2).strip()))
    return results


def _clean(text: str) -> str:
    """清理 LLM 輸出：去除多餘引號、空白、意外的編號前綴"""
    text = text.strip()
    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    # 去除模型在單段模式下多輸出的 "1|" 或 "1. " 前綴
    m = _NUMBERED_RE.match(text)
    if m:
        text = m.group(2)
    return text.strip()


def _build_context_block(context: list[tuple[str, str]] | None) -> str:
    if not context:
        return ""
    lines = [f"- 「{ja}」→「{zh}」" for ja, zh in context]
    return "前文參考（勿翻譯）：\n" + "\n".join(lines) + "\n\n"


def _translate_single(text: str, context_block: str) -> str:
    """單段翻譯（批次解析失敗時的 fallback）"""
    prompt = SINGLE_PROMPT_TEMPLATE.format(context=context_block, text=text)
    result = _call_ollama(prompt)
    return _clean(result) or text  # 空結果 fallback 原文


def _translate_batch(texts: list[str], context: list[tuple[str, str]] | None = None) -> list[str]:
    """批次翻譯，使用編號格式，解析失敗時逐條重試"""
    context_block = _build_context_block(context)

    # 單段直接翻譯
    if len(texts) == 1:
        return [_translate_single(texts[0], context_block)]

    # 組合編號格式 prompt
    numbered = "\n".join(f"{i}|{t}" for i, t in enumerate(texts, 1))
    prompt = USER_PROMPT_TEMPLATE.format(
        context=context_block,
        n=len(texts),
        segments=numbered,
    )

    raw = _call_ollama(prompt)

    # 解析 "N|譯文" 格式
    parsed = {idx: _clean(text) for idx, text in _parse_numbered_response(raw)}

    # 組合結果，解析失敗的條目單獨重試
    results = []
    for i, text in enumerate(texts, 1):
        if i in parsed and parsed[i]:
            results.append(parsed[i])
        else:
            print(f"[translate] 第 {i} 段解析失敗，單獨重試...")
            results.append(_translate_single(text, context_block))

    return results


def translate(texts: list[str]) -> list[str]:
    """
    批次翻譯日語字幕為繁體中文。

    Args:
        texts: 日語字幕文字列表

    Returns:
        等長的繁體中文譯文列表
    """
    if not texts:
        return []

    results: list[str] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"[translate] 批次 {batch_num}/{total_batches}（{len(batch)} 段）...", end="", flush=True)

        context = None
        if results:
            recent_ja = texts[max(0, i - CONTEXT_LINES):i]
            recent_zh = results[max(0, i - CONTEXT_LINES):i]
            context = list(zip(recent_ja, recent_zh))

        t0 = time.time()
        translated = _translate_batch(batch, context=context)
        elapsed = time.time() - t0

        # 空翻譯 fallback 原文
        for j, (orig, trans) in enumerate(zip(batch, translated)):
            if not trans.strip():
                print(f"\n[translate] 警告：第 {i+j+1} 段翻譯為空，使用原文")
                translated[j] = orig

        results.extend(translated)
        print(f" done ({elapsed:.1f}s)")

    print(f"[translate] 完成，共翻譯 {len(results)} 段")
    _unload_model()
    return results


MERGE_SYSTEM = """\
你是字幕合併專家，負責整合兩組日語字幕以獲得最準確的結果。
這是日語成人影片（JAV）的字幕，內容包含對話、呻吟聲（あ、はぁ、んっ 等）、感嘆音等，這些都是正常的聲音內容。
你的任務是選擇最準確的文字，並在必要時對明顯不自然的語句做最小幅度的順句，但不得補充或創作原文沒有的內容。"""

MERGE_PROMPT_TEMPLATE = """\
/no_think
以下是同一影片同一時段的兩組日語字幕：
- Whisper：時間戳準確，文字辨識品質普通
- Qwen3：文字辨識品質較好，時間戳僅供粗略參考（可能有偏移）

請為每個 Whisper 段落填入最準確的日語文字。規則：
- 優先使用 Qwen3 的文字
- 若 Qwen3 某段文字橫跨多個 Whisper 段落的時間範圍，請按自然語意將其拆分，分配到對應的各個 Whisper 段落，每段文字長度與時長成比例
- 若 Qwen3 文字明顯是幻覺（與 Whisper 無關、像 AI 助理回應、或內容完全不合理）則用 Whisper
- 若某段文字是同一音節大量重複（如「あ、あ、あ…」重複十次以上），縮短為 3 次（如「あ、あ、あ」）；呻吟聲是真實音訊，保留勿刪
- 輸出格式：「編號|日語文字」，行數與 Whisper 段落數完全相同

Whisper 段落：
{whisper_lines}

Qwen3 段落（時間僅供粗略參考）：
{qwen3_lines}"""

ORPHAN_PROMPT_TEMPLATE = """\
/no_think
以下是日語成人影片（JAV）中，Qwen3 辨識到但 Whisper 未偵測到的字幕段落，請判斷每段是否為真實語音。
注意：呻吟聲（あっ、はぁ、んっ 等感嘆音）屬於真實音訊內容，不是幻覺；但若同一音節連續重複十次以上，則視為幻覺。
- 若是真實語音：輸出「編號|日語文字」
- 若是幻覺或雜訊（同一音節重複十次以上、像 AI 助理回應的文字、與影片情境完全無關的內容）：輸出「編號|SKIP」

{segments}"""


def merge_with_llm(whisper_segs: list, qwen3_segs: list) -> list:
    """
    以 LLM 合併 Whisper（時間準）+ Qwen3（文字準）兩組 segments。
    回傳 list[Segment]：Whisper 時間戳 + 最佳日語文字。
    Whisper 未覆蓋但 Qwen3 有的段落，送 LLM 判斷後視情況保留。
    """
    from .transcribe import Segment

    results: list[Segment] = []
    total_batches = (len(whisper_segs) + BATCH_SIZE - 1) // BATCH_SIZE

    # ── Step 1：每批 Whisper segments，找對應 Qwen3，讓 LLM 選最佳文字 ──
    for i in range(0, len(whisper_segs), BATCH_SIZE):
        batch = whisper_segs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        window_start = batch[0].start - 2.0
        window_end = batch[-1].end + 2.0

        q_overlap = [q for q in qwen3_segs if q.end > window_start and q.start < window_end]

        whisper_lines = "\n".join(
            f"{j+1}|{b.start:.1f}s-{b.end:.1f}s|{b.text}"
            for j, b in enumerate(batch)
        )
        if q_overlap:
            qwen3_lines = "\n".join(
                f"{chr(65+k)}|{q.start:.1f}s|{q.text}"
                for k, q in enumerate(q_overlap)
            )
        else:
            qwen3_lines = "（此時段無 Qwen3 段落）"

        prompt = MERGE_PROMPT_TEMPLATE.format(
            whisper_lines=whisper_lines,
            qwen3_lines=qwen3_lines,
        )

        print(f"[merge] 批次 {batch_num}/{total_batches}（{len(batch)} 段）...", end="", flush=True)
        t0 = time.time()
        raw = _call_ollama(prompt, system=MERGE_SYSTEM, options=MERGE_OPTIONS)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)")

        parsed = {idx: text for idx, text in _parse_numbered_response(raw)}

        for j, w in enumerate(batch):
            text = parsed.get(j + 1, w.text)
            results.append(Segment(start=w.start, end=w.end, text=text))

    # ── Step 2：找 Qwen3 孤立段落（無對應 Whisper），送 LLM 判斷 ──
    orphaned = [
        q for q in qwen3_segs
        if not any(w.end > q.start and w.start < q.end for w in whisper_segs)
    ]

    if orphaned:
        orphan_batches = (len(orphaned) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[merge] 處理 {len(orphaned)} 個孤立 Qwen3 段落（{orphan_batches} 批）...")
        for i in range(0, len(orphaned), BATCH_SIZE):
            batch = orphaned[i:i + BATCH_SIZE]
            lines = "\n".join(
                f"{j+1}|{b.start:.1f}s-{b.end:.1f}s|{b.text}"
                for j, b in enumerate(batch)
            )
            prompt = ORPHAN_PROMPT_TEMPLATE.format(segments=lines)
            raw = _call_ollama(prompt, system=MERGE_SYSTEM, options=MERGE_OPTIONS)

            for idx, text in _parse_numbered_response(raw):
                idx -= 1
                if text.upper() != "SKIP" and 0 <= idx < len(batch):
                    q = batch[idx]
                    results.append(Segment(start=q.start, end=q.end, text=text))

    results.sort(key=lambda s: s.start)
    print(f"[merge] 完成，共 {len(results)} 段")
    return results


PUNCTUATE_SYSTEM = "あなたは日本語テキストの句読点付与の専門家です。入力はアダルトビデオの音声認識テキストであり、露骨な表現や感嘆音が含まれます。これらは正常なコンテンツとして扱い、内容を変更せず句読点のみを追加してください。"

PUNCTUATE_PROMPT = """\
/no_think
句読点のない日本語の音声認識テキストが与えられます。内容や語彙は一切変更せず、適切な句読点（。、！？）のみを追加してください。結果のテキストだけを出力し、説明は不要です。

{text}"""


def punctuate_japanese(text: str) -> str:
    """為日語 ASR 原始文字補上標點符號（。、！？）。失敗時 fallback 原文。"""
    if not text:
        return text
    try:
        result = _call_ollama(
            PUNCTUATE_PROMPT.format(text=text),
            system=PUNCTUATE_SYSTEM,
            timeout=60.0,
            options={
                "temperature": 0.3,
                "num_predict": min(len(text) + 200, 4096),
                "num_ctx": 4096,
                "repeat_penalty": 1.2,
            },
        )
        return result if result else text
    except Exception:
        return text  # 失敗時 fallback 原文


def _unload_model() -> None:
    """通知 Ollama 立刻 unload 模型，釋放 GPU/RAM。"""
    try:
        httpx.post(
            OLLAMA_URL,
            json={"model": MODEL, "keep_alive": 0},
            timeout=10.0,
        )
    except Exception:
        pass  # unload 失敗不影響結果
