# 文档入口

本目录是“基于 MCP 的本地化个人知识库助手系统”的开发规范入口。Codex 或 oh-my-codex 在进行开发、测试或文档调整前，必须先读取本文件，再按阅读顺序读取其余规范。

本文件是唯一动态入口，负责说明当前第二轮开发目标、阅读顺序、计划入口、允许范围和禁止范围。其他文档只描述稳定实现规范，不承载动态目标、过程记录或运行记录。

## 当前开发目标

当前目标是推进第二轮开发标准：在第一轮已完成 `kb init -> kb add -> kb index -> kb search -> kb ask` 最小 RAG 闭环的基础上，更新正式 docs，使后续实现能够按规范引入：

- 文本清洗 pipeline；
- 传统 chunking，并通过配置项解耦；
- 本地 `bge-base-en-v1.5` embedding；
- Python 进程内直接调用本地 embedding 模型；
- Chroma 本地向量数据库；
- OpenAI-compatible LLM API；
- 完整 RAG 流程：clean → chunk → embed → vector store → vector retrieval → LLM answer；
- Recall@k、端到端响应延迟、embedding 耗时、检索耗时、LLM token 消耗；
- `metrics.jsonl` 与 `sessions.jsonl` 职责边界；
- `.kcontext/config.json` 配置扩展边界；
- 第一轮 `.kcontext` 数据兼容与重建边界；
- embedding、Chroma、LLM API 不可用时的 fallback 边界。

当前执行计划入口：`.omx/plans/docs-round2-upgrade-ralplan-final.md`。

## 必读顺序

1. `docs/README.md`：确认当前目标、阅读顺序和边界。
2. `docs/project_spec.md`：确认项目定位、系统闭环、功能范围和非目标。
3. `docs/architecture.md`：确认分层架构、本地与外部边界、MCP 边界、模型与向量库边界。
4. `docs/rag_pipeline.md`：确认 RAG 管线顺序、策略配置、索引与查询流程、fallback 关系。
5. `docs/data_contracts.md`：确认数据字段、配置项、持久化记录、指标事件和回溯不变量。
6. `docs/module_interfaces.md`：确认模块职责、输入输出、依赖方向和错误面。
7. `docs/testing_and_evaluation.md`：确认测试与评价口径、指标定义、采集方式和证据等级。
8. `.omx/plans/docs-round2-upgrade-ralplan-final.md`：确认第二轮 docs 升级计划和 Step8 审计要求。

## 当前允许工作范围

- 更新现有六份正式 docs。
- 新增且只新增 `docs/rag_pipeline.md`。
- 将 RAG 管线、配置项、Chroma、embedding、LLM、metrics、Recall@k 规范写入对应 SSOT 文档。
- 保持 README 作为唯一动态入口。
- 保持非 README 文档为稳定实现规范。

## 当前禁止工作范围

- 不修改 `src/`。
- 不修改 `tests/`。
- 不修改 `pyproject.toml`。
- 不修改 `.omx/plans/docs-round2-upgrade-ralplan-final.md`。
- 不新增计划外 docs 文件。
- 不写代码。
- 不实现 OCR、DOCX、摘要生成、多用户、云同步、chunking 方法对比实验、本地 App UI、后台进程管理。
- 不写真实 API key。
- 不写真实性能结果或运行记录。
- 不设置固定数值通过条件。

## 全局硬约束摘要

- 项目形态：面向单个个人用户的本地知识库助手。
- 支持文档类型：PDF、Markdown、TXT。
- 不支持文档类型：DOCX 及其他未列明格式。
- 实现语言：Python。
- embedding 默认模型：本地 `bge-base-en-v1.5`。
- embedding 接入：Python 进程内直接调用本地模型。
- 向量数据库：Chroma，本地持久化。
- LLM 接入：OpenAI-compatible API。
- API key：只允许从环境变量读取，不落盘。
- 检索默认模式：vector retrieval。
- fallback：保留 keyword search fallback 与 no-LLM grounded synthesis fallback。
- 回答约束：必须基于 retrieved chunks，并输出 sources。

## 文档职责索引

| 文件 | 职责 | SSOT 范围 |
|---|---|---|
| `README.md` | 唯一动态入口、阅读顺序、当前目标、允许与禁止范围 | 当前目标 / 优先级 / 计划入口 |
| `project_spec.md` | 项目定位、系统闭环、范围、非目标、高层成功原则 | 项目范围 / 非目标 |
| `architecture.md` | 分层架构、依赖方向、本地与外部边界、MCP 边界 | 架构层次 / 边界 |
| `rag_pipeline.md` | RAG 管线、策略配置、索引与查询流程、fallback 关系 | 流程顺序 / 策略使用 |
| `data_contracts.md` | 数据对象、配置项、持久化记录、指标事件、不变量 | 字段 / 数据结构 / 不变量 |
| `module_interfaces.md` | 模块职责、输入输出、依赖方向、错误面 | 模块职责 / I-O / 错误面 |
| `testing_and_evaluation.md` | 测试与评价口径、指标定义、证据等级 | 评价方法 / 采集口径 |

## 变更规则

1. 当前目标、阅读顺序或计划入口变化，只修改 `docs/README.md`。
2. 项目范围、非目标或高层成功原则变化，修改 `docs/project_spec.md`。
3. 架构层次、依赖方向、本地/外部边界变化，修改 `docs/architecture.md`。
4. RAG 流程顺序、策略配置使用方式、fallback 关系变化，修改 `docs/rag_pipeline.md`。
5. 字段、配置项、持久化记录结构或不变量变化，修改 `docs/data_contracts.md`。
6. 模块职责、输入输出或错误面变化，修改 `docs/module_interfaces.md`。
7. 指标定义、采集口径或证据等级变化，修改 `docs/testing_and_evaluation.md`。
8. 若文档之间出现冲突，按各领域 SSOT 归属裁决；无法裁决时记录冲突并由用户决断。
