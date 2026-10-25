# 基于 RAG 的心理健康知识智能问答系统

这是一个用于学习和面试演示的轻量级 RAG 项目。系统面向心理健康科普与一般知识问答，不提供疾病诊断、治疗替代或药物处方。

## 当前进度

- [x] 创建并验证 Conda 环境
- [x] 建立最小 Python 项目骨架
- [x] 实现单个 UTF-8 TXT 文档读取与基础元数据生成
- [x] 实现保守、可重复的单文档文本清洗
- [x] 实现清洗后正文的确定性文档去重与审计记录
- [ ] 实现文本分块与片段元数据
- [ ] 建立 Embedding 与 FAISS 索引
- [ ] 实现检索、生成、引用与拒答
- [ ] 建立评估集和 Streamlit 演示

## 环境

```powershell
conda env create -f environment.yml
conda activate mental-health-rag
```

当前机器已经创建该环境，因此日常开发只需执行激活命令。

## 验证项目骨架

```powershell
pytest
ruff check .
```

## 当前模块：TXT 文档读取

`load_text_document(path)` 读取一个非空 UTF-8 `.txt` 文件，返回包含正文与来源元数据的 LangChain `Document`。当前模块只负责读取，不执行清洗、去重或分块。

## 当前模块：单文档清洗

`clean_document(document)` 对 LangChain `Document` 执行保守且确定性的清洗：使用 NFC 规范化 Unicode，统一换行和横向空白，移除常见零宽字符，并合并多余空行。函数返回新文档，不修改输入对象；原始 `document_id` 会被保留，清洗后的正文另有哈希值用于后续去重。

## 当前模块：确定性文档去重

`deduplicate_documents(documents)` 按清洗后正文的 SHA-256 保留第一次出现的文档，并返回稳定顺序的唯一文档和逐条重复审计记录。函数会重新计算哈希以识别过期元数据，不执行可能误删不同结论的语义去重。

## 安全边界

本项目仅提供心理健康科普信息。对于自伤、自杀或其他紧急危险信号，系统应进入专门的安全响应流程，引导用户联系当地紧急服务、危机热线或可信任的人，不能输出危险操作建议。

