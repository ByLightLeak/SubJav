# SubJav

本地日語影片字幕自動生成工具。全程離線，不上傳任何資料。

**流程：** 影片 → ASR 轉錄日語 → LLM 翻譯繁體中文 → 輸出 SRT

---

## 需求

- Apple Silicon Mac（使用 MLX 加速）
- Python 3.11+
- [Ollama](https://ollama.com/) 並已拉取模型：`ollama pull qwen3:14b`
- ffmpeg：`brew install ffmpeg`

---

## 安裝

```bash
git clone https://github.com/ByLightLeak/SubJav.git
cd SubJav
uv sync
```

---

## 設定

```bash
cp subjav.toml.example subjav.toml
```

編輯 `subjav.toml`，至少設定 `video_dir`：

```toml
[defaults]
# 影片根目錄（必填）
video_dir = "/path/to/your/videos"

# ASR 後端：whisper / qwen3 / hybrid（預設 whisper）
backend = "hybrid"

# true → 只產 SRT，不嵌入影片
# false → 產 SRT 並嵌入軟字幕
srt_only = true

# true → 嵌入硬字幕（燒錄進畫面，任何播放器都看得到）
# false → 嵌入軟字幕（可在播放器切換顯示）
hard_sub = false
```

`subjav.toml` 已加入 `.gitignore`，不會被 commit。

---

## 使用

```bash
# 自動掃描 video_dir 下所有影片，逐一處理
uv run subjav

# 給目錄名，自動找裡面的影片
uv run subjav fc2-ppv-1234567/

# 指定單一影片（相對於 video_dir）
uv run subjav fc2-ppv-1234567/fc2-ppv-1234567_720p.mp4

# 指定絕對路徑
uv run subjav /path/to/video.mp4
```

**常用選項：**

| 選項 | 說明 |
|------|------|
| `--backend` | ASR 後端：`whisper` / `qwen3` / `hybrid` |
| `--srt-only` | 只產 SRT，不嵌入影片 |
| `--hard-sub` | 嵌入硬字幕（燒錄進畫面） |
| `--force` / `-f` | 強制重新轉錄，刪除已有快取 |
| `--output-dir` | 指定輸出目錄 |

CLI 選項優先於 `subjav.toml` 的設定值。

---

## ASR 後端

| 後端 | 速度 | 時間戳 | 日語辨識 |
|------|------|--------|----------|
| `whisper` | 快 | 準確 | 普通 |
| `qwen3` | 中 | 粗略 | 好 |
| `hybrid` | 慢 | 準確 | 好 |

`hybrid` 以 Whisper 時間戳為準，文字取 Qwen3 結果，再用 LLM 合併，品質最佳。

---

## 輸出檔案

影片在 `fc2-ppv-1234/fc2-ppv-1234_720p.mp4` 的情況下：

```
fc2-ppv-1234/
  fc2-ppv-1234_720p.mp4                    # 原始影片（不動）
  fc2-ppv-1234_720p_hybrid.srt             # 中日雙語字幕
  fc2-ppv-1234_720p.wav                    # 音軌暫存（轉錄用）
  fc2-ppv-1234_720p_whisper_segments.json  # Whisper 轉錄快取
  fc2-ppv-1234_720p_qwen3_segments.json    # Qwen3 轉錄快取
```

重複執行時會跳過已有的快取，只補缺少的步驟。
