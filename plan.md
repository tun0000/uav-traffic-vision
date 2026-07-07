# uav-traffic-vision 專案計畫

## Context

求職作品集專案 #4：**無人機視角物件偵測與交通流量分析**。用 Ultralytics YOLO26 在 VisDrone2019-DET 上訓練空拍小目標偵測，SAHI 切片推論為技術主菜，偵測 + ByteTrack + 虛擬計數線的車流統計為 README 主視覺，邊緣部署討論對接台灣無人機產業職缺（無人機巡檢、海巡、智慧交通）。

已確認的前提與決策（2026-07 上網查證）：
- **YOLO26 已於 2026-01 正式釋出**：n/s/m/l/x 五種尺寸；支援 detect/seg/pose/OBB；NMS-free 端到端（一對一 head，輸出固定 `(N, 300, 6)`）；COCO mAP：yolo26n 40.9 / yolo26s 48.6。實作時仍先查官方文件確認最低版本與 API（CLAUDE.md 約定）。
- **ultralytics 內建 `VisDrone.yaml`**：自動下載共 2.3 GB（train 6,471 / val 548 / test-dev 1,610 張），轉換腳本內建、自動略過 ignored regions；10 類：pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor。
- **SAHI 官方支援 YOLO26**（`AutoDetectionModel.from_pretrained(model_type="ultralytics")`），ultralytics docs 有專門的 SAHI guide。
- **VisDrone VID/MOT 影片子集不在 VisDrone.yaml 內**，需從 VisDrone GitHub 的雲端連結手動下載。VisDrone 授權為學術研究用途，README 與 HF model card 必須註明出處與非商用性質。
- 環境沿用系列慣例（同專案 2 的 CLAUDE.md）：WSL2 + 4090 本機 smoke test、Colab Pro 正式訓練（L4）、uv 鎖依賴、資料放 WSL `~/datasets/`、checkpoint 存 Drive、所有報告數字必須來自實際執行、上傳 GitHub/HF 前先確認 repo 名稱與可見性。

## 關鍵技術判斷（相對原始提案的補強）

1. **三段式小目標敘事**：imgsz=640 baseline → imgsz=1024 對照 → SAHI 切片。README 用同一套指標呈現三段遞進，講清楚「解析度就是空拍小目標的命脈」。
2. **YOLO26 端到端 head 每圖上限 300 偵測**：VisDrone 密集場景單圖可達數百目標，300 上限會直接壓 recall——這是 SAHI 的**第二個賣點**（每個 slice 各有 300 上限，等效放寬全圖上限）。實作時查證 e2e 模式下 `max_det` 能否調高；Step 1 的 EDA 先統計「單圖目標數 > 300 的圖有幾張」，把截斷影響量化寫進分析。
3. **統一評估底座**：把 VisDrone val 轉成 COCO JSON，「直接推論」與「SAHI 推論」都輸出 COCO 格式預測、統一用 pycocotools 評估 → 同一套 mAP / AR / 面積分桶（COCO small/medium/large + 自訂 tiny < 16²）數字可直接並排對照。`model.val()` 另跑一次作為標準 ultralytics 數字的 sanity 對照（兩套數字都進 README，註明評估器差異）。
4. **虛擬計數線的前提是機位固定**：無人機若在移動，畫面上的固定線沒有物理意義。VisDrone MOT 要挑**懸停／近似固定機位**的路口或路段序列；找不到就 fallback 公開授權空拍車流影片（Pexels/Pixabay 類明確允許使用的，附出處與授權）。
5. **車流統計只計車輛類**（car, van, truck, bus, motor, tricycle, awning-tricycle）；pedestrian/people/bicycle 偵測照畫、但不入車流計數，stats.json 分開列。
6. **資料集 EDA 是 README 資產**：bbox 面積分桶直方圖 + 每圖目標數分布，放 README 用來 motivate SAHI 與 1024 對照，不只是內部檢查。

## 專案結構

```
4_無人機視角偵測與車流分析/   (GitHub repo: uav-traffic-vision)
├── plan.md                  # 本計畫（核准後落地到 repo）
├── CLAUDE.md                # 沿用系列工作約定
├── README.md / LICENSE(MIT) / .gitignore / .env.example
├── pyproject.toml           # uv；deps: ultralytics, sahi, pycocotools, opencv-python,
│                            #   numpy, pandas, python-dotenv；Phase 2 再加 onnx,
│                            #   onnxruntime, gradio, tensorrt(WSL)
├── scripts/
│   ├── dataset_stats.py     # Phase 1：EDA → reports/dataset_stats.md + 直方圖
│   ├── make_subset.py       # Phase 1：smoke test 小子集 + subset yaml
│   ├── evaluate.py          # Phase 2：val + per-class AP + 面積分桶（COCO eval）
│   ├── sahi_compare.py      # Phase 2：直接 vs SAHI 對照（指標 + 並排圖）
│   ├── traffic_count.py     # Phase 2：track + 計數線 → stats.json + 標註影片
│   └── export_benchmark.py  # Phase 2：ONNX / TensorRT FP16 匯出與 benchmark
├── notebooks/train_yolo26_visdrone_colab.ipynb
├── app/                     # Phase 2：Gradio demo（HF Space, CPU ONNX）
├── reports/                 # 統計、sanity、結果圖、GIF（小檔進 git）
└── weights/                 # gitignored；Phase 2 放 yolo26s_visdrone_640.pt 等（.pt/.onnx/.engine）
資料實體：WSL ~/datasets/VisDrone/（ultralytics settings 的 datasets_dir 指過去；不進 git）
```

## Phase 1（本機 + 產出 notebook，v1 實作範圍）

### Step 0：初始化專案
- `git init`；`.gitignore`（datasets、weights/、runs/、*.pt/*.onnx/*.engine、.env、影片大檔）；MIT LICENSE；README skeleton；`.env.example`；CLAUDE.md；本計畫存為 repo 根目錄 `plan.md`。
- WSL 內 uv 建 venv 鎖依賴。動工前先查 ultralytics 官方文件確認 YOLO26 / VisDrone.yaml / tracker API 現況。

### Step 1：資料下載 + EDA
- 確認 ultralytics `datasets_dir` 指向 `~/datasets`，用 VisDrone.yaml 觸發自動下載與格式轉換（2.3 GB）。
- `dataset_stats.py` → `reports/dataset_stats.md`：train/val 各類 instance 數、影像解析度分布、**bbox 面積分桶直方圖**（tiny/small/medium/large）、**每圖目標數分布（含 >300 張數統計）**。統計結果輸出給使用者看。

### Step 2：4090 smoke test（WSL）
- `make_subset.py` 抽 ~300 train / 100 val + subset data.yaml。
- `yolo26n.pt`、1 epoch、imgsz=640：loss 正常下降、val 跑通、無 NaN；predict 2–3 張 val 圖疊圖人工確認。

### Step 3：`notebooks/train_yolo26_visdrone_colab.ipynb`（Runtime → Run all 一鍵跑完）
- Cells：GPU check → mount Drive → `git clone` 本 repo → pip install → VisDrone 自動下載到 **Colab 本地磁碟**（非 Drive，I/O 快）→ 訓練 → 驗證 → 產物存 Drive。
- 訓練設定：`yolo26s.pt`、imgsz=640、epochs=100 + patience=20、seed 固定、augmentation 走預設（空拍場景不強加 flipud/rotation，v1 不過度調參）。
- `project=` 指到 Drive 路徑 → checkpoint 自動落 Drive；附 `resume=True` 斷線續訓說明 cell；結束後 best.pt / results.csv 複製到 Drive 固定路徑。
- **notebook 末尾「imgsz=1024 對照實驗」可選 cell**：先跑 2 epoch @1024 實測單 epoch 時間，換算總時數與 Colab compute units（L4 費率設為參數），**印出估算後停住**；需手動把 `RUN_1024 = False` 改 True 才會真的開訓。1024 權重另存 Drive 不覆蓋 640 版。

### Step 4：git commit（英文、一個里程碑一個 commit）
**Phase 1 到此停止，等使用者去 Colab 訓練、把權重放進 weights/（本專案命名慣例：`yolo26s_visdrone_640.pt`，避免跨專案都叫 best.pt 撞名）。**

## Phase 2（weights/yolo26s_visdrone_640.pt 就位後）

### Step 5：評估 `evaluate.py`
- `model.val()`：mAP50 / mAP50-95、per-class AP → README 結果表。
- COCO eval 路徑：val GT 轉 COCO JSON + 預測轉 COCO format → pycocotools：AP_small/medium/large + 自訂 tiny 桶、AR@100 → **小目標表現分析**段落。
- 輸出代表性預測疊圖（密集場景、小目標場景各數張）到 `reports/figures/`。
- 若 1024 對照有跑：640 vs 1024 同表對照。

### Step 6：SAHI 切片推論 `sahi_compare.py`（主菜）
- SAHI `get_sliced_prediction`：初始參數 640×640 slice、overlap 0.2，小範圍掃描（2–3 組）取合理設定。
- 直接推論 vs SAHI：同一套 COCO eval 對照 **mAP 與 recall**，重點看 tiny/small 桶的差異；量化「300 上限截斷」在兩種模式下的影響。
- 同一張圖「直接 vs SAHI」**並排可視化**數張（挑小目標密集的畫面）→ README 核心圖。
- 附推論耗時對照（SAHI 慢多少倍，誠實列出 trade-off）。

### Step 7：車流分析 `traffic_count.py`
- 影片來源：優先 VisDrone MOT 子集挑**固定機位**車流序列；不可得 → 公開授權空拍車流影片（附出處與授權聲明）。
- `model.track(tracker="bytetrack.yaml", persist=True)` + 自寫計數線模組：track ID 跨線偵測、雙向計數、per-class 統計（僅車輛類入計數）。
- 輸出：`stats.json`（各類別、各方向、總計）+ 標註影片（框、ID、軌跡尾巴、計數板）→ 壓成 README 主 GIF。
- 人工驗收：目測數一段影片的車數與 stats.json 對照。

### Step 8：邊緣部署 `export_benchmark.py`
- export ONNX（end-to-end）與 TensorRT FP16 engine（4090 上建）。
- Benchmark：ONNX-CPU / TensorRT-FP16-4090，batch=1、imgsz=640、warmup 10 + 100 次，報 mean/p50/p95。
- README 部署段：**「遷移到 Jetson（Orin 級）機載電腦的路徑與預期瓶頸」**——TensorRT engine 需在目標機重建、記憶體/功耗預算、預期吞吐量級；**誠實註明所有量測在桌機 4090/CPU 上做**。
- 寫入 YOLO26 NMS-free 對機載部署的意義：固定輸出形狀、無 NMS plugin 相依、延遲確定性、量化路徑更簡單。

### Step 9：發佈
- HF Hub 上傳權重 + model card（VisDrone 出處、學術用途授權註明；**repo 名稱與可見性先跟使用者確認**）。
- Gradio demo（`app/`）：圖片偵測版、CPU ONNX（`YOLO("best.onnx")` 路徑）、**SAHI on/off toggle**（預設關，附 CPU 切片推論較慢的警語）、內建 2–3 張範例圖；可部署免費 Space。影片追蹤用 README GIF 展示即可。
- README 完成：動機（台灣無人機產業應用：巡檢、海巡、智慧交通）、資料集 EDA 圖、640/1024/SAHI 三段結果表、SAHI 並排圖、車流 GIF、benchmark 表、Jetson 遷移段、重現步驟、HF 連結、授權聲明。

## Phase 3（選做，使用者已於 2026-07-07 明確說開始）

延續同一空拍領域，換一種標註型態：DOTA-v1.0 的旋轉框（OBB）。定位是「補充展示」而非第二個完整專案——不重做 SAHI／車流／邊緣部署那一整套，只走「資料 → 訓練 → 輕量評估」，證明同一套 YOLO26 pipeline 換一個 task head、換一種標註方式也能順利跑通。

已確認的前提（2026-07 上網查證 ultralytics 官方文件與原始碼，含直接讀 `split_dota.py` 原始碼確認函式簽名）：
- **DOTA-v1.0**：15 類（plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool）；2,806 張影像、188,282 個標註（train 1,411 / val 458 / test 937，test 無標籤）；ultralytics `DOTAv1.yaml` 自動下載（~2GB）。學術用途授權（commercial use prohibited）。
- **標籤格式**：`class_id x1 y1 x2 y2 x3 y3 x4 y4`（4 個角點，正規化座標）——與 VisDrone 的 `class_id xc yc w h` 完全不同，EDA／smoke test 都要對應調整（角點轉 (w,h,angle) 沿用 `ultralytics.utils.ops.xyxyxyxy2xywhr` 同一套 `cv2.minAreaRect` 邏輯，確保跟模型實際訓練時的角度定義一致）。
- **原始影像過大，無法直接餵給 YOLO**：本專案實測長邊 421–13,383px、中位數 ~2100px、85%+ 超過 1024px（見 `reports/dota_stats.md`）。官方要求先用 `ultralytics.data.split_dota.split_trainval()` 切成重疊的 1024×1024 tile（函式預設 `crop_size=1024, gap=200, rates=(1.0,)`；官方文件範例另外展示 multiscale `rates=[0.5,1,1.5]` 但那是 3 倍前處理與訓練資料量的加強版，本專案先用函式預設的單一尺度控制 Colab 成本，multiscale 留作 README 中的可選延伸）。`split_dota` 依賴 `shapely`（已 `uv add shapely` 補齊，避免重演 tensorrt 那次的 pip/uv 依賴漂移)。
- **YOLO26-obb**：`yolo26n-obb.pt` ~ `yolo26x-obb.pt`，輸出角度正規化到 `[-pi/4, 3pi/4)`；`model.val()` 內建 rotated-IoU 版 mAP（不需要像 VisDrone 那樣另外接 pycocotools）；SAHI 官方文件未提及支援 OBB，本階段不做 SAHI 對照。
- 沿用主線的權重命名慣例：`yolo26s_dota_1024.pt`（資料集+解析度命名，避免跨專案泛用檔名）。

### Step 10：DOTA EDA（本機）— 已完成
`scripts/dota_stats.py`：觸發官方自動下載（沿用 VisDrone 的 DATASETS_DIR 凍結陷阱 fix：`settings.update` 後另開 subprocess），對 train/val 原始（未切片）影像統計：類別分佈、影像解析度分布（motivate 切片的必要性）、旋轉框轉 (w,h,angle) 後的面積分桶（沿用 VisDrone 同一套 tiny/small/medium/large 門檻方便跨資料集對照）、角度分布（量化有多少比例的框其實不是軸對齊）、長寬比分布（量化細長物件如 bridge/harbor 的存在）。輸出 `reports/dota_stats.md` + 4 張圖。

### Step 11：本機 split_dota 切片 + 4090 smoke test
`scripts/prepare_dota_obb.py`：對完整 train+val（1,869 張）跑一次 `split_trainval`（本機負責資料前處理，符合 CLAUDE.md 分工），輸出到 `~/datasets/DOTAv1-split/`；再從切好的 tile 中抽樣（300 train / 100 val，沿用 VisDrone smoke test 的抽樣數）建 `DOTAv1_obb_subset` + yaml。用 `yolo26n-obb.pt`、1 epoch、imgsz=1024 在小子集上驗證：loss 正常下降、val 算得出 OBB mAP、predict 疊圖目視確認框確實是旋轉的（非純軸對齊）。

### Step 12：`notebooks/train_yolo26obb_dota_colab.ipynb`（Runtime → Run all 一鍵跑完）
- Colab 內重新下載 DOTAv1 原始資料 + 重新跑一次 `split_trainval`（Colab 本地磁碟，同 Phase 1 VisDrone「不依賴本機上傳、Colab 自己重下」的慣例）。
- 訓練設定：`yolo26s-obb.pt`、imgsz=1024（tile 尺寸本身就是 1024，不需要像主線那樣另外比較 640 vs 1024）、epochs=100、patience=20、seed 固定；batch 依偵測到的 GPU 動態調整（A100/L4，沿用主線 1024 實驗學到的保守值起手：A100 24 / L4 8，因為 OBB head 在 1024 下的實際顯存足跡未知，先保守再視第一個 epoch 的 GPU_mem 決定要不要調高）。
- 結束後 best.pt 複製到 Drive 固定路徑，檔名沿用 `yolo26s_dota_1024` 命名慣例。
- **此 notebook 完成後 Phase 3 本機端工作停止**，等使用者去 Colab 訓練、把權重放進 `weights/`。

### Step 13（權重就位後，選做）：輕量評估
- `model.val()` 直接拿 OBB 內建 mAP50 / mAP50-95（rotated IoU），不必另建 pycocotools pipeline。
- 挑 2–3 張代表性影像（密集船隻／機場、細長橋樑）疊圖展示旋轉框（用 ultralytics 內建的 OBB plot）。
- README 加一小節「另一種標註型態」，附 EDA 圖 + 訓練結果 + 疊圖，定調為補充展示而非第二個主線章節。

### 風險與備案（Phase 3 專屬）
- **DOTA-v1.0 官方下載失效** → 使用者手動下載官方連結，格式仍走 ultralytics 內建 loader。
- **split_dota 本機跑太久或磁碟不夠** → 本機磁碟餘裕 900GB+，實測風險低；真的太久可退而求其次只切 train+val 各一個子集。
- **OBB 在 1024 的實際顯存需求超乎預期**（比照主線 1024 實驗 A100 93% VRAM 的教訓）→ Colab notebook 預設用保守 batch，不做本機沒驗證過的激進值。

## 風險與備案

- **VisDrone 自動下載失效** → 官方 GitHub 手動連結；轉換仍用 ultralytics 內建腳本。
- **MOT 子集不可得或無固定機位序列** → 公開授權空拍車流影片（明確授權、附出處）。
- **e2e 模式 max_det 無法調高且 >300 目標圖占比高** → 量化截斷影響寫進分析，以 SAHI 補足；必要時該實驗改用 NMS head 模式跑並註明。
- **SAHI 與 YOLO26 e2e 輸出相容性問題** → 官方已有 guide，風險低；若有問題改以標準 head 跑 SAHI 並在 README 註明。
- **Colab L4 太慢** → 降 epochs（patience 會提早停）或換 A100；1024 對照本來就先估時再決定。
- **Space CPU 上 SAHI 太慢** → toggle 預設關 + 警語 + 限制切片數。

## Verification（各階段驗收）

- Step 1：dataset_stats.md 的張數/類別與官方數字量級一致（6,471/548、10 類）；直方圖肉眼合理。
- Step 2：1 epoch loss 下降、val 跑通、疊圖無明顯錯位。
- Step 3：本機以小子集逐 cell 驗證流程；`jupyter nbconvert --to script` 可解析。
- Step 5–6：兩套評估（model.val vs pycocotools）的整體 mAP 量級一致；SAHI 並排圖肉眼可見小目標增益。
- Step 7：stats.json 與人工目測計數一致（抽一段驗）；GIF 可正常播放且 < 10 MB。
- Step 8：ONNX/TRT 推論結果與 .pt 抽樣比對（同圖偵測數與框位置一致）；benchmark 數字寫進 README 前重跑一次確認穩定。
- 全程：README 所有數字來自實際執行輸出，嚴禁估計（CLAUDE.md 約定）。
