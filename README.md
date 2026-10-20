# 基于 RAG 的心理健康知识智能问答系统

这是一个用于学习和面试演示的轻量级 RAG 项目。系统面向心理健康科普与一般知识问答，不提供疾病诊断、治疗替代或药物处方。

## 当前进度

- [x] 创建并验证 Conda 环境
- [x] 建立最小 Python 项目骨架
- [ ] 实现单一格式文档读取
- [ ] 实现清洗、去重和分块
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

## 安全边界

本项目仅提供心理健康科普信息。对于自伤、自杀或其他紧急危险信号，系统应进入专门的安全响应流程，引导用户联系当地紧急服务、危机热线或可信任的人，不能输出危险操作建议。

