"""Gradio UI shared by the RAG02 notebook and local interface validation."""

import gradio as gr

from rag_core import SOURCE_RECORD_URL, SOURCE_TITLE, answer_question

CUSTOM_CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.sensor-card { border: 1px solid #c8dfdf; border-radius: 14px; padding: 15px;
  background: linear-gradient(135deg, #f0faf8, #eff6fc); line-height: 1.8; }
"""

EXAMPLES = [
    "濕度高時 PM2.5 感測器讀值變高，能直接判定污染變嚴重嗎？",
    "什麼是共置（collocation）？為什麼要把感測器和參考儀器放在一起？",
    "如果要比較校門口和操場的空氣品質，開始量測前應先規劃哪些事？",
    "今天中原大學的 PM2.5 是多少？",
]


def create_demo(store, manifest, client=None):
    with gr.Blocks(title="AirScope SensorLab｜空品感測器避坑教練") as demo:
        gr.Markdown("""# 🌬️ AirScope SensorLab｜空品感測器避坑教練
**先找文件，再談結論。** 阿澈陪你查手冊、看證據，分清楚「感測器讀到了什麼」與「能不能據此下判斷」。
""")
        gr.Markdown(
            f"📖 **資料：** [{SOURCE_TITLE}（2022）]({SOURCE_RECORD_URL})　"
            f"**PDF：** {manifest['pdf_pages']} 頁　**向量片段：** {manifest['chunks']} 筆　"
            f"**維度：** {manifest['dimension']}"
        )
        gr.Markdown("這份資料是技術手冊的文字快照，未連接即時空品資料；來源頁碼使用 PDF 頁序。")
        with gr.Row():
            with gr.Column(scale=1, min_width=250):
                gr.HTML("""<div class="sensor-card"><b>使用方式</b><br>
                ① 選擇回覆風格<br>② 描述你的量測問題<br>③ 送出後展開文件證據核對引用<br><br>
                <b>查不到就說查不到。</b><br>相似度用來排序資料，不等於答案正確率。</div>""")
                style = gr.Radio(["快速避坑", "實驗設計教練"], value="快速避坑", label="回覆風格")
                k = gr.Slider(minimum=2, maximum=6, value=4, step=1, label="檢索片段數 k")
                preview = gr.Checkbox(value=client is None, label="純檢索預覽（不呼叫 LLM）")
                key_status = "已讀取 Groq 金鑰；送出後才呼叫模型。" if client else "尚未讀取 Groq；可先操作純檢索預覽。"
                gr.Markdown(key_status)
                clear = gr.Button("🧹 清除本次對話", variant="secondary")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="與阿澈查文件", height=430)
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="例如：下雨後讀值變高，是污染還是量測誤差？",
                        container=False, scale=5,
                    )
                    send = gr.Button("查文件並送出", variant="primary", scale=1)
                status = gr.Markdown("準備完成。請先輸入問題。")
        with gr.Accordion("📚 本輪文件證據（對照 [S1] 等引用）", open=False):
            evidence = gr.Markdown("送出問題後，這裡會顯示原文片段、PDF 頁碼與檢索相似度。")
        gr.Markdown("### 一鍵帶入測試題")
        with gr.Row():
            buttons = [gr.Button(text) for text in ["🌧️ 濕度與讀值", "🔬 共置校正", "🏫 校園量測規劃", "🧪 文件外問題"]]
        for button, example in zip(buttons, EXAMPLES):
            button.click(lambda text=example: text, outputs=message)

        def respond(question, history, top_k, selected_style, preview_only):
            return answer_question(store, question, history, top_k, selected_style, client, preview_only)

        inputs = [message, chatbot, k, style, preview]
        outputs = [message, chatbot, evidence, status]
        message.submit(respond, inputs=inputs, outputs=outputs, api_name="rag_chat")
        send.click(respond, inputs=inputs, outputs=outputs, api_name="rag_chat_send")
        clear.click(
            lambda: ("", [], "送出問題後，這裡會顯示本輪原文證據。", "已清除對話。"),
            outputs=outputs,
            api_name="clear_chat",
        )
    return demo


def launch_demo(demo, share=True, debug=True):
    return demo.queue().launch(
        share=share, debug=debug,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="sky"),
        css=CUSTOM_CSS,
    )
