# 基于 RAG 的心理健康知识智能问答系统

这是一个用于学习和面试演示的轻量级 RAG 项目。系统面向心理健康科普与一般知识问答，不提供疾病诊断、治疗替代或药物处方。

## 当前进度

- [x] 创建并验证 Conda 环境
- [x] 建立最小 Python 项目骨架
- [x] 实现单个 UTF-8 TXT 文档读取与基础元数据生成
- [x] 实现保守、可重复的单文档文本清洗
- [x] 实现清洗后正文的确定性文档去重与审计记录
- [x] 建立并真实抓取权威心理健康中文语料库 v1
- [x] 实现单篇及批量文档的确定性中文分块与片段元数据
- [ ] 实现分块结果 JSONL 落盘与可复现构建命令
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

## 当前模块：权威语料库 v1

`data/sources/corpus_v1.json` 是冻结的来源清单。当前 16 篇资料全部来自世界卫生组织或中国政府网转载的国家卫生健康委文件，覆盖心理健康基础、压力、抑郁、焦虑、双相情感障碍、精神分裂症、儿童青少年、老年人、工作场所、认知健康、酒精相关风险、自杀预防、危机识别和 12356 心理援助热线。

构建过程只抽取指定正文容器，不进行摘要或同义改写；每篇资料必须通过至少 3 个按正文顺序设置的人工核验短语。原始文件使用 SHA-256 留痕，清洗后执行精确去重。完整原文不提交到公开 Git 仓库，来源清单和审计报告可以提交。

```powershell
$env:PYTHONPATH = "src"
python -m mental_health_rag.corpus
```

如需重新访问官方来源并刷新原始缓存：

```powershell
$env:PYTHONPATH = "src"
python -m mental_health_rag.corpus --refresh
```

输出：

- `data/raw/corpus_v1/`：原始网页缓存，仅供本机审计，已被 Git 忽略。
- `data/processed/corpus_v1.jsonl`：清洗后的 LangChain `Document` 序列，已被 Git 忽略。
- `data/reports/corpus_v1_report.json`：来源哈希、字符数、主题分布和语义核验结果。

## 当前模块：中文文本分块

`split_document(document, config)` 使用 LangChain 的
`RecursiveCharacterTextSplitter`（递归字符文本分割器），优先沿换行和中文句末标点切分，
必要时逐级回退到更细边界。正文只从清洗后原文中截取，不执行摘要或同义改写。

`split_documents(documents, config)` 对一批文档应用同一配置，保持文档顺序和片段顺序，
返回全部片段以及逐文档统计。批处理中若发现重复 `document_id` 会立即失败，避免不同文档
产生含义不清或重复的片段身份；错误信息保留输入位置，便于定位坏数据。

首版基线参数是 `chunk_size=500`、`chunk_overlap=80`，长度单位为 Python 字符数。
每个片段继承来源元数据，并新增 `chunk_id`、`parent_document_id`、`chunk_index`、
`chunk_count`、`start_index`、`end_index`、内容哈希和分块配置等字段。

```powershell
conda activate mental-health-rag
pytest tests/test_chunking.py -q
```

在当前 16 篇真实语料上的只读回归产生 109 个片段：最短 127 字、中位数 452 字、
平均 421.11 字、最长 498 字；超出 500 字的片段为 0，109 个片段 ID 均唯一，
且全部可以按起止位置精确回指清洗正文。这些是当前语料与基线参数下的真实结果，
不代表参数已经最优；检索效果仍需在后续评估集上验证。

批量接口对同一语料再次得到 16 篇输入和 109 个片段，每篇产生 3 至 13 个片段，
中位数为 7；文档顺序保持不变，所有片段 ID 仍然唯一。

## 安全边界

本项目仅提供心理健康科普信息。对于自伤、自杀或其他紧急危险信号，系统应进入专门的安全响应流程，引导用户联系当地紧急服务、危机热线或可信任的人，不能输出危险操作建议。

