# 模块接口规格

## 文档职责

本文档定义模块职责、输入输出、可依赖对象、不可承担职责和错误面。字段结构以 `data_contracts.md` 为准，流程顺序以 `rag_pipeline.md` 为准。

## parser

- 职责：将 PDF、Markdown、TXT 转换为 DocumentBlock。
- 输入：Document metadata、文件路径、文件类型。
- 输出：DocumentBlock 列表、解析状态。
- 可依赖：本地文件系统、格式解析工具、DocumentBlock 数据契约。
- 不可承担：文本清洗策略、chunking、embedding、检索、LLM 调用。
- 错误面：文件不存在、格式不支持、PDF 文本不可提取、解析结果为空。

## cleaner

- 职责：按 `cleaning_profile` 对 DocumentBlock 文本执行规则化清洗。
- 输入：DocumentBlock 列表、cleaning 配置。
- 输出：清洗后的 DocumentBlock 列表或等价清洗结果。
- 可依赖：DocumentBlock 数据契约、清洗规则配置。
- 不可承担：生成 chunk、生成 embedding、修改来源文档、改变 block 回溯语义。
- 错误面：清洗后无有效文本、block 字段缺失、非法 block 顺序。

## chunking_strategy

- 职责：定义可替换 chunking 策略。
- 输入：清洗后的 DocumentBlock 列表、chunking 配置。
- 输出：Chunk 列表。
- 可依赖：Chunk 与 DocumentBlock 数据契约。
- 不可承担：读取原始文件、调用 embedding、写入 Chroma、检索排序。
- 错误面：无法生成可追溯 chunk、chunk 文本为空、block_ids 缺失。

## chunking_service

- 职责：按 `chunking_strategy` 调度具体策略，并持久化 chunks。
- 输入：DocumentBlock 列表、chunking 配置、知识库根目录。
- 输出：Chunk 列表、写入状态。
- 可依赖：chunking_strategy、chunk repository。
- 不可承担：向量生成、LLM 调用、评价统计。
- 错误面：未知策略、chunk 超出约束、写入失败。

## embedding_provider

- 职责：为 chunk 文本生成 embedding。
- 输入：文本列表、`embedding_model`、`embedding_device`。
- 输出：向量列表、embedding metadata、耗时信息。
- 可依赖：本地 `bge-base-en-v1.5` 模型、模型加载工具、设备配置。
- 不可承担：chunking、检索排序、写入 Chroma、生成回答。
- 错误面：模型不可用、device 不可用、维度不匹配、文本为空、批处理失败。

## vector_store

- 职责：封装向量库写入、查询、删除和重建能力。
- 输入：VectorRecord、查询向量、collection 配置。
- 输出：SearchResult 原始候选、写入状态、索引状态。
- 可依赖：Chroma、本地持久化目录、VectorRecord 数据契约。
- 不可承担：生成 embedding、文本清洗、生成 LLM 回答。
- 错误面：Chroma 不可用、collection metadata 不兼容、向量维度不匹配、持久化失败。

## retrieval_service

- 职责：根据 `retrieval_mode` 返回 Top-K 相关 chunk。
- 输入：query、top_k、retrieval_mode、知识库状态。
- 输出：SearchResult 列表、空结果状态、实际 retrieval mode。
- 可依赖：embedding_provider、vector_store、keyword index、Chunk 数据契约。
- 不可承担：生成最终回答、伪造引用、直接检索整篇文档替代 chunk。
- 错误面：索引缺失、embedding 不可用、Chroma 不可用、查询为空、召回为空。

## llm_client

- 职责：封装 OpenAI-compatible API 调用。
- 输入：用户问题、retrieved chunks 上下文、系统提示词、`llm_base_url`、`llm_model`、环境变量 API key。
- 输出：模型文本、TokenUsage、调用状态。
- 可依赖：OpenAI-compatible HTTP API、运行时环境变量。
- 不可承担：读取全量知识库、执行检索、生成 sources、保存会话。
- 错误面：API key 缺失、base_url 不可用、模型错误、网络失败、响应超时、usage 缺失。

## answer_service

- 职责：编排检索结果到回答输出的生成过程。
- 输入：question、SearchResult 列表、LLM 配置、fallback 配置。
- 输出：AnswerResult。
- 可依赖：llm_client、source mapper、TokenUsage 数据契约。
- 不可承担：文档解析、索引构建、直接访问 Chroma 底层状态。
- 错误面：检索结果为空、LLM 不可用、上下文过长、sources 不可追溯。

## source_mapper

- 职责：将 SearchResult 映射为 AnswerResult sources。
- 输入：SearchResult、Chunk、Document metadata。
- 输出：Source 列表。
- 可依赖：Chunk、Source、DocumentBlock 回溯关系。
- 不可承担：生成模型回答、修改 chunk 文本、伪造来源位置。
- 错误面：chunk 缺失、source_doc_name 缺失、block_ids 不可解析、quote 不可生成。

## metrics_collector

- 职责：记录指标事件。
- 输入：operation、duration、retrieval metadata、TokenUsage、关联 session/query。
- 输出：MetricsEvent、写入状态。
- 可依赖：MetricsEvent 数据契约、本地 JSONL 存储。
- 不可承担：改变业务结果、替代会话存储、判断功能是否通过。
- 错误面：时间记录不可用、写入失败、token usage 来源不合法、关联对象缺失。

## index_service

- 职责：编排 `kb index` 的 clean、chunk、embed、upsert Chroma 与指标记录。
- 输入：知识库根目录、配置对象、待索引文档范围。
- 输出：索引构建摘要、chunk 写入状态、vector 写入状态、指标引用。
- 可依赖：cleaner、chunking_service、embedding_provider、vector_store、metrics_collector。
- 不可承担：导入原始文件、回答用户问题、调用 LLM。
- 错误面：blocks 缺失、清洗后无有效文本、embedding 不可用、Chroma 不可用、配置不兼容。

## knowledge_base_service

- 职责：作为应用层门面，协调 init、add、index、search、ask 等用例。
- 输入：命令参数、知识库根目录、配置对象。
- 输出：命令级结果对象或错误状态。
- 可依赖：各应用服务与基础设施接口。
- 不可承担：具体模型推理、具体向量库实现细节、UI 渲染。
- 错误面：知识库未初始化、配置缺失、依赖不可用、持久化失败。

## CLI

- 职责：提供 `kb init`、`kb add`、`kb index`、`kb search`、`kb ask` 的薄入口。
- 输入：命令行参数。
- 输出：用户可读摘要、错误信息、必要 JSON 输出。
- 可依赖：knowledge_base_service。
- 不可承担：业务规则、字段契约定义、底层索引实现。
- 错误面：参数缺失、路径不存在、命令失败、依赖不可用。

## 依赖规则

- CLI 只能调用应用层服务。
- 应用层编排流程，不直接绑定具体厂商 SDK。
- 基础设施层实现 Chroma、模型加载、HTTP API、本地存储。
- 领域层不依赖基础设施层。
- module_interfaces 引用 data_contracts，不重定义字段默认值。

## fallback 接口要求

- embedding_provider 失败时，retrieval_service 可选择 keyword fallback。
- vector_store 失败时，retrieval_service 不得返回伪造向量结果。
- llm_client 失败时，answer_service 可选择 no-LLM grounded synthesis。
- fallback 状态必须体现在返回对象或指标事件中。
