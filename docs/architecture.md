# 架构规格

## 架构目标

系统采用本地优先的分层架构，将文档处理、索引构建、检索、模型调用、指标采集和交互入口解耦。架构必须支持 CLI 入口、MCP 能力组织边界和个人本地知识库的数据安全要求。

## 分层结构

### 表现层

表现层负责接收用户命令或界面操作，并展示结果。表现层不得承载核心业务规则，不得直接访问 Chroma、embedding 模型或 LLM API。

### 应用层

应用层编排用例流程，包括导入、索引、检索、问答和指标记录。应用层负责协调 parser、cleaner、chunker、embedding provider、vector store、retriever、LLM client 和 metrics collector。

### 领域层

领域层定义核心对象和不变量，包括 Document、DocumentBlock、Chunk、SearchResult、AnswerResult、Source、MetricsEvent 等。领域层不依赖具体外部库。

### 基础设施层

基础设施层封装本地文件存储、Chroma、embedding 模型加载、OpenAI-compatible API 调用和本地配置读取。基础设施层实现必须通过应用层接口暴露能力。

## 核心数据流

系统核心数据流为：

1. 文档导入为 Document metadata。
2. parser 输出 DocumentBlock。
3. cleaner 按 `cleaning_profile` 输出清洗后块。
4. chunker 按 `chunking_strategy` 输出 Chunk。
5. embedding provider 基于 `embedding_model` 和 `embedding_device` 生成向量。
6. vector store 将向量与 metadata upsert 到 Chroma。
7. retriever 按 `retrieval_mode` 召回 chunk。
8. answer service 组装上下文并调用 LLM client。
9. citation/source mapper 返回可追溯 sources。
10. metrics collector 记录耗时、token 与检索指标事件。

## 本地处理边界

以下能力必须在本地完成：

- 文档持久化；
- 文档解析；
- 文本清洗；
- chunking；
- embedding 模型调用；
- Chroma 向量库持久化；
- 检索；
- 指标事件写入。

外部 API 只用于 LLM 回答生成，不得作为默认文档存储或索引存储位置。

## embedding 架构边界

- 默认 embedding 模型为本地 `bge-base-en-v1.5`。
- embedding 通过 Python 进程内模型调用完成。
- embedding provider 位于基础设施层，通过应用层接口调用。
- `embedding_device` 由配置提供，架构不得绑定特定硬件加速设备。
- 模型不可用、模型加载失败或 device 不可用时，应用层必须接收明确错误，并允许检索或问答走已定义 fallback。
- 未来将 embedding 能力拆为固定本地服务时，不应改变应用层调用契约。

## Chroma 架构边界

- Chroma 是本地向量数据库，位于基础设施层。
- Chroma 数据必须持久化到本地目录。
- 持久化目录由 `chroma_persist_dir` 配置决定。
- Chroma collection metadata 用于区分 embedding 模型、chunking 策略、清洗配置和索引版本。
- Chroma 不替代 `chunks.jsonl`；`chunks.jsonl` 仍作为 chunk 契约和回溯依据。
- Chroma 不承担文档解析、文本清洗、chunking 或 LLM 调用职责。

## LLM API 架构边界

- LLM client 封装 OpenAI-compatible API。
- API key 只允许从环境变量读取。
- `llm_base_url` 与 `llm_model` 可由配置提供，并允许运行时环境变量覆盖。
- 运行时覆盖不得反向写入 `.kcontext/config.json`。
- LLM API 只接收用户问题、retrieved chunks 和必要系统提示词。
- LLM API 不得默认接收全量文档、全库内容或与问题无关的材料。
- LLM API 不可用时，问答服务应进入 no-LLM grounded synthesis fallback 或返回明确不可用状态。

## MCP 边界

MCP 用于组织解析、模型、检索、评价等能力边界。MCP 适配层不得扩展为通用智能体调度层、插件商店或多主体协作平台。MCP 适配器只负责标准化能力接入，不拥有领域数据契约。

## 配置加载边界

- `.kcontext/config.json` 是本地知识库配置持久化位置。
- 配置字段语义以 `data_contracts.md` 为准。
- 配置读取应由基础设施层封装。
- 应用层只消费配置对象，不直接解析散落字段。
- API key 不属于 `.kcontext/config.json` 持久化内容。

## `.kcontext` 兼容与重建边界

- 既有 `metadata.jsonl`、`blocks.jsonl`、`chunks.jsonl` 应保持可读取。
- 缺少第二代配置字段时，应通过默认配置补齐运行语义。
- 若 chunk 内容、清洗配置、chunking 策略、embedding 模型、embedding 维度或索引版本变化，应触发索引重建或增量更新。
- 重建不得破坏原始文档 metadata 与 DocumentBlock 回溯链。

## fallback 边界

- embedding 不可用：不得写入新的向量索引；可保留 keyword search fallback。
- Chroma 不可用：vector retrieval 不可用；search 可回退到 keyword retrieval，并明确 retrieval mode。
- LLM API 不可用：ask 可回退到 no-LLM grounded synthesis，并明确依据不足或模型不可用。
- fallback 不得伪装成默认成功路径；输出或指标事件必须能区分实际路径。

## 禁止的架构耦合

- 表现层直接调用 LLM API。
- 表现层直接操作 Chroma。
- retriever 直接读取原始全文替代 chunk 检索。
- LLM client 直接读取本地知识库。
- metrics collector 反向影响业务输出。
- Chroma metadata 替代 `data_contracts.md` 的字段定义。
