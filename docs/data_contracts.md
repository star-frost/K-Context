# 数据契约规格

## 文档职责

本文档是数据字段、配置项、持久化结构、不变量和回溯关系的唯一规范源。模块职责不在本文档定义，流程顺序不在本文档定义，评价公式不在本文档定义。

## 通用约定

- 时间字段使用 ISO 8601 字符串。
- 标识字段必须在同一知识库内稳定唯一。
- JSONL 文件每行保存一个完整 JSON 对象。
- 缺失的可选字段使用 `null` 或省略策略，但必须在读取时可区分未知与空值。
- 所有可追溯对象必须能关联到 source document。

## `.kcontext/config.json`

`.kcontext/config.json` 是知识库本地配置持久化文件。字段语义、默认值语义和合法值范围由本文档定义。

### 配置字段

| 字段 | 含义 | 约束 |
|---|---|---|
| `embedding_model` | embedding 模型标识 | 默认语义为本地 `bge-base-en-v1.5` |
| `embedding_device` | embedding 运行设备 | 可为 `auto`、`cpu` 或运行环境支持的设备名；不得绑定特定硬件加速设备 |
| `vector_store_type` | 向量库类型 | 默认语义为 `chroma` |
| `chroma_persist_dir` | Chroma 本地持久化目录 | 相对知识库根目录或绝对路径均须可解析 |
| `llm_base_url` | OpenAI-compatible API base URL | 不包含 API key |
| `llm_model` | LLM 模型名 | 不包含密钥信息 |
| `chunking_strategy` | chunking 策略名 | 默认语义为传统 chunking |
| `cleaning_profile` | 文本清洗规则集名 | 默认语义为基础清洗规则 |
| `retrieval_mode` | 检索模式 | 支持 vector、keyword；hybrid 为可组合模式名 |
| `top_k` | 默认召回数量 | 正整数 |

### 环境变量覆盖

- `KCONTEXT_LLM_API_KEY` 是 LLM API key 唯一来源，不得落盘。
- `KCONTEXT_LLM_BASE_URL` 可以作为 `llm_base_url` 的运行时覆盖值。
- `KCONTEXT_LLM_MODEL` 可以作为 `llm_model` 的运行时覆盖值。
- 运行时覆盖不反向写入 `.kcontext/config.json`。
- 指标事件可记录有效配置来源，但不得记录密钥。

## Document

Document 表示导入的本地文件。

必备字段：

| 字段 | 含义 |
|---|---|
| `document_id` | 文档唯一标识 |
| `file_name` | 文件名 |
| `file_type` | pdf、markdown、txt |
| `source_path` | 原始文件路径或导入来源 |
| `storage_ref` | 本地存储引用 |
| `created_at` | 导入时间 |
| `status` | 处理状态 |

不变量：

- DOCX 与未列明格式不得产生成功 Document。
- `document_id` 是 block、chunk、source 的回溯入口。

## DocumentBlock

DocumentBlock 是解析后的统一中间表示。

必备字段：

| 字段 | 含义 |
|---|---|
| `block_id` | block 唯一标识 |
| `document_id` | 来源文档标识 |
| `order` | 文档内顺序 |
| `text` | block 文本 |
| `block_type` | 段落、标题、列表、代码块、页文本等类型 |
| `page` | 页码或等价位置，可为空 |
| `heading_path` | 标题路径，可为空 |
| `metadata` | 格式相关扩展信息 |

不变量：

- 清洗不得破坏 `block_id`、`document_id`、`order` 的回溯语义。
- 空文本 block 可被标记或过滤，但处理结果必须可解释。

## Chunk

Chunk 是检索的基本对象。

必备字段：

| 字段 | 含义 |
|---|---|
| `chunk_id` | chunk 唯一标识 |
| `source_doc_id` | 来源文档标识 |
| `source_doc_name` | 来源文档名 |
| `block_ids` | 来源 block 标识列表 |
| `text` | chunk 文本 |
| `order` | chunk 顺序 |
| `page_start` | 起始页或等价位置 |
| `page_end` | 结束页或等价位置 |
| `chunking_strategy` | 生成该 chunk 的策略 |
| `cleaning_profile` | 生成该 chunk 前使用的清洗配置 |
| `chunk_text_hash` | chunk 文本 hash |
| `created_at` | 创建时间 |

不变量：

- 每个 chunk 必须可回溯到至少一个 DocumentBlock。
- chunk 文本变化必须导致 `chunk_text_hash` 变化。
- 不同 chunking 策略生成的数据必须可区分。

## EmbeddingRecord

EmbeddingRecord 表示 chunk 对应的 embedding 缓存信息。

必备字段：

| 字段 | 含义 |
|---|---|
| `embedding_id` | embedding 记录标识 |
| `chunk_id` | 对应 chunk |
| `chunk_text_hash` | embedding 输入文本 hash |
| `embedding_model` | embedding 模型 |
| `embedding_dim` | 向量维度 |
| `created_at` | 生成时间 |
| `status` | 生成状态 |

重建触发条件：

- chunk 文本变化；
- `chunk_text_hash` 变化；
- `embedding_model` 变化；
- `embedding_dim` 变化；
- `chunking_strategy` 变化；
- `cleaning_profile` 变化；
- `index_version` 变化。

## VectorRecord

VectorRecord 表示写入向量库的 chunk 向量及 metadata。

必备 metadata：

| 字段 | 含义 |
|---|---|
| `chunk_id` | chunk 标识 |
| `source_doc_id` | 来源文档标识 |
| `source_doc_name` | 来源文档名 |
| `block_ids` | 来源 block 列表 |
| `chunking_strategy` | chunking 策略 |
| `cleaning_profile` | 清洗配置 |
| `embedding_model` | embedding 模型 |
| `embedding_dim` | 向量维度 |
| `chunk_text_hash` | chunk 文本 hash |
| `index_version` | 索引版本 |
| `created_at` | 写入时间 |

不变量：

- Chroma 中的向量 metadata 不得缺失回溯字段。
- 不同 embedding 模型或向量维度的记录不得被无标识混合解释。
- 不同 chunking 策略或清洗配置的记录不得被无标识混合解释。

## Chroma Collection Metadata

Chroma collection metadata 至少应包含：

| 字段 | 含义 |
|---|---|
| `embedding_model` | collection 对应 embedding 模型 |
| `embedding_dim` | collection 向量维度 |
| `chunking_strategy` | collection 对应 chunking 策略 |
| `cleaning_profile` | collection 对应清洗配置 |
| `index_version` | 索引版本 |
| `created_at` | collection 创建时间 |

## SearchResult

SearchResult 表示检索返回项。

必备字段：

| 字段 | 含义 |
|---|---|
| `chunk_id` | 命中的 chunk |
| `source_doc_id` | 来源文档标识 |
| `source_doc_name` | 来源文档名 |
| `score` | 相关性分数 |
| `retrieval_mode` | 实际检索模式 |
| `block_ids` | 来源 block 列表 |
| `text` | 命中文本或片段 |
| `metadata` | 其他可追溯信息 |

## AnswerResult

AnswerResult 表示问答结果。

必备字段：

| 字段 | 含义 |
|---|---|
| `answer` | 回答正文 |
| `evidence_level` | 证据等级 |
| `sources` | Source 列表 |
| `token_usage` | TokenUsage，可为空但需标记来源 |
| `metrics_ref` | 关联指标事件，可为空 |

`evidence_level` 只允许：

- `证据不足`
- `基本充分`
- `充分`

## Source

Source 表示回答依据。

必备字段：

| 字段 | 含义 |
|---|---|
| `chunk_id` | 来源 chunk |
| `source_doc_id` | 来源文档标识 |
| `source_doc_name` | 来源文档名 |
| `score` | 检索分数 |
| `block_ids` | 来源 block 列表 |
| `page_start` | 起始页或等价位置 |
| `page_end` | 结束页或等价位置 |
| `quote` | 可展示片段 |

## TokenUsage

TokenUsage 表示 LLM 调用 token 记录。

必备字段：

| 字段 | 含义 |
|---|---|
| `prompt_tokens` | 输入 token 数，可为空 |
| `completion_tokens` | 输出 token 数，可为空 |
| `total_tokens` | 总 token 数，可为空 |
| `source` | 统计来源 |

`source` 只允许：

- `api_usage`：API 响应提供 usage；
- `estimated`：本地估算；
- `unavailable`：不可获得且未估算。

## MetricsEvent

MetricsEvent 表示指标事件，建议写入 `.kcontext/metrics.jsonl`。

必备字段：

| 字段 | 含义 |
|---|---|
| `event_id` | 指标事件标识 |
| `event_type` | index、embedding、retrieval、ask、llm_call、evaluation 等 |
| `timestamp` | 记录时间 |
| `duration_ms` | 事件耗时 |
| `operation` | 操作名称 |
| `retrieval_mode` | 检索模式，可为空 |
| `top_k` | 召回数量，可为空 |
| `embedding_model` | embedding 模型，可为空 |
| `vector_store_type` | 向量库类型，可为空 |
| `token_usage` | TokenUsage，可为空 |
| `related_session_id` | 关联会话，可为空 |
| `related_query_id` | 关联问题，可为空 |

## `sessions.jsonl` 与 `metrics.jsonl`

| 文件 | 职责 |
|---|---|
| `sessions.jsonl` | 保存会话、问题、回答、sources、交互历史 |
| `metrics.jsonl` | 保存性能、耗时、token、检索与评价相关测量事件 |

不变量：

- 会话历史不得混写为指标事件。
- 指标事件不得替代 answer 或 source 记录。
- 指标结构只定义记录格式，不表达运行数据结论。

## Recall Ground Truth

Recall 评价标注结构应支持：

| 字段 | 含义 |
|---|---|
| `query_id` | 问题标识 |
| `question` | 问题文本 |
| `relevant_doc_ids` | 相关文档标识列表 |
| `relevant_chunk_ids` | 相关 chunk 标识列表，可为空 |
| `evidence_text` | 人工标注依据文本，可为空 |
| `evidence_text_hash` | 依据文本 hash，可为空 |
| `source_span` | 文档内位置，可为空 |
| `label_level` | doc-level 或 evidence-level |

约束：

- ground truth 必须来自人工标注。
- 不得使用模型自评替代人工标注。
- chunk id 不稳定时，应通过文档、位置、证据文本或 hash 保留可比性。
