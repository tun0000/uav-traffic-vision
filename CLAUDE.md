# 專案環境與工作約定

## 硬體與系統
- Windows 11 + WSL2 Ubuntu，所有指令都在 WSL 內執行。本機 GPU：RTX 4090（24GB VRAM）。
- 模型訓練預設在 Google Colab（Pro）執行：你要產生可以「Runtime → Run all」一鍵跑完的 .ipynb。
- 本機負責：資料前處理、smoke test、推論、匯出、benchmark、demo、文件。

## 帳號與金鑰
- GitHub 與 Hugging Face 都已登入（huggingface-cli 有 write token）。
- API 金鑰（GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY）放在 repo 根目錄 .env，一律用 python-dotenv 讀取。.env 必須在 .gitignore 內，絕不硬編碼、絕不 commit 金鑰。

## 開發約定
- 動手整合任何套件前，先上網查該套件（ultralytics、sahi、gradio 等）目前的官方文件確認 API 用法，不要憑記憶猜。
- 先小後大：先在本機 4090 用小子集 + 1 epoch 做 smoke test 驗證整條流程，確定沒問題才產出完整 Colab notebook。
- 依賴版本用 pyproject.toml（uv）鎖定；訓練固定 random seed；重要腳本都要有 argparse 與 docstring。
- uv venv 放 WSL `~/venvs/uav-traffic-vision`（`UV_PROJECT_ENVIRONMENT` 指過去），避免 /mnt/c 的 I/O 瓶頸；資料集放 WSL `~/datasets/`。
- 資料集、權重、影片等大檔一律不進 git（.gitignore 先寫好）；權重發佈走 Hugging Face Hub。
- 每完成一個里程碑就 git commit（訊息用英文、簡潔、一件事一個 commit）。
- 所有報告中的數字必須來自實際執行結果，嚴禁捏造或估計指標。
- 每個 repo 完成時必備：README.md（專案動機、結果表格、demo GIF 佔位、重現步驟、HF 連結）、LICENSE、資料集出處與授權聲明。
- 上傳任何東西到 GitHub / Hugging Face 之前，先跟我確認 repo 名稱與可見性。
