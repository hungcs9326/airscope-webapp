# 資料與程式來源紀錄

## 文件

- 名稱：The Enhanced Air Sensor Guidebook
- 機構：U.S. Environmental Protection Agency
- 版本：2022；報告編號 EPA/600/R-22/213
- 官方說明：https://cfpub.epa.gov/si/si_public_record_report.cfm?Lab=CEMM&dirEntryId=356426
- 官方 PDF：https://cfpub.epa.gov/si/si_public_file_download.cfm?Lab=CEMM&p_download_id=545843
- 官方概覽：https://www.epa.gov/air-sensor-toolbox/how-use-air-sensors-air-sensor-guidebook
- 下載日期：2026-10-30
- PDF：195 頁、9,724,392 bytes
- PDF SHA-256：`a6b5ea68a2e4dd4b55674c2b2ff78dc209549fd23f3ae5371bc25ab0c70446e3`
- ZIP SHA-256：`697fb25c46f4bcf37df15621a36bcae20752446cefb6256ba463f742e8a09468`

文字取自原 PDF，不是由 AI 自行撰寫的知識來源。圖表未 OCR，旋轉文字與表格可能抽取不完整；引用採實際 PDF 頁序。

EPA 著作權說明：https://www.epa.gov/web-policies-and-procedures/epa-disclaimers 。依其文件說明供非商業科學與教育用途使用，保留來源，不宣稱整份文件為 CC0，也不暗示 EPA 認可本作品。

## 課堂

- 使用者提供直播：https://www.youtube.com/watch?v=jSNrM7hjhek
- 標題：【TAICA_生成式AI：文字與圖像生成的原理與實務】7. 讓AI看懂你的世界
- 教師講義入口：https://yenlung.me/1151GenAI
- 第 7 週講義檔案 ID：`1DPmDRSQ1mZPZDeGUUHAPOB4GlUdWFCbp`
- 教師範例：https://github.com/yenlung/AI-Demo
- `【Demo06a】RAG01_打造向量資料庫.ipynb` blob SHA：`57a435b25f8335512dffb0111c043dd007161ce5`
- `【Demo06b】RAG02_打造_RAG_系統.ipynb` blob SHA：`e061b47fc19fda90598c82c0eb9c3f4ca6d5d542`

影片未提供字幕；本次檢查閱讀相對應完整講義和原始程式，不將其描述成完整觀看直播。

## 技術依據

- E5：https://huggingface.co/intfloat/multilingual-e5-small
- 固定 revision：`614241f622f53c4eeff9890bdc4f31cfecc418b3`
- 模型授權：MIT；此授權僅指嵌入模型，不代表 EPA 文件授權
- Groq 模型：https://console.groq.com/docs/models
- LangChain FAISS：https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
- 套件版本：`requirements-colab.txt`；索引建立環境紀錄：`manifest.json`

需連網安裝套件、下載嵌入模型與呼叫 Groq。API 金鑰由每位執行者在 Colab Secrets 自行設定，不放入任何公開檔案。
