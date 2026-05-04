# 模块与接口边界

## 模块总览

系统采用集中式模块边界说明，不为每个模块拆分独立文档。每个模块必须有明确职责、输入对象、输出对象、可依赖对象和错误面。

| 模块 | SSOT 职责 |
|---|---|
| document_ingest | 文档导入与元数据登记 |
| document_parse | PDF、Markdown、TXT 到 IR 的转换 |
| document_clean | IR 文本清洗与噪声处理 |
| chunking | 基于 IR 的结构感知切块 |
| retrieval | Chunk 索引与 Top-K 检索 |
| llm_service | OpenAI-compatible API 封装 |
| citation_mapper | 检索片段到引用对象的映射 |
| chat_service | 问答流程编排与会话关联 |
| ui_desktop | 桌面端界面展示与用户交互 |
| mcp_adapters | 能力接入与服务组织边界 |
| evaluation | 测试测量与证据判定支持 |

## document_ingest

- 职责：接收用户选择的本地文件，校验格式，创建 Document 元数据，记录处理状态。
- 输入：本地文件路径、用户导入请求。
- 输出：Document 元数据、导入成功或失败结果。
- 可依赖：文件系统接口、Document 存储接口。
- 不可承担：解析正文、生成 IR、建立索引、调用 LLM。
- 错误面：文件不存在、格式不支持、读取失败、重复导入策略冲突。

## document_parse

- 职责：将 PDF、Markdown、TXT 解析为 DocumentBlock IR。
- 输入：Document 元数据、本地文件引用。
- 输出：DocumentBlock 列表、解析状态。
- 可依赖：具体解析器、OCR 工具接口、DocumentBlock 数据契约。
- 不可承担：索引建立、问答生成、UI 展示。
- 错误面：解析失败、文本不可提取、格式损坏、OCR 工具不可用。

## document_clean

- 职责：对 DocumentBlock 文本执行清洗规则，去除噪声并保留结构信息。
- 输入：DocumentBlock 列表。
- 输出：清洗后的 DocumentBlock 列表。
- 可依赖：数据契约中的字段语义和清洗规则。
- 不可承担：跨页或跨标题无依据合并、切块长度策略、检索排序。
- 错误面：清洗后无有效文本、结构字段缺失、非法 block_type。

## chunking

- 职责：基于 IR 结构边界和长度约束生成 Chunk。
- 输入：清洗后的 DocumentBlock 列表。
- 输出：Chunk 列表。
- 可依赖：DocumentBlock 与 Chunk 数据契约。
- 不可承担：直接读取原始文件、直接调用 LLM、检索排序。
- 错误面：无法回溯 block_ids、Chunk 超出长度上限、来源信息缺失。

## retrieval

- 职责：为 Chunk 建立本地索引，并根据用户问题返回 Top-K 相关 Chunk。
- 输入：Chunk 列表、用户问题或查询表示。
- 输出：检索结果列表、相关性信息、空结果状态。
- 可依赖：Embedding 服务、向量索引或本地检索库、Chunk 数据契约。
- 不可承担：生成最终回答、伪造引用、检索整篇文档替代 Chunk。
- 错误面：索引缺失、索引未更新、查询表示失败、召回为空。

## llm_service

- 职责：封装 OpenAI-compatible API 调用，为应用层提供稳定的 LLM 接口。
- 输入：用户问题、检索得到的 Chunk 上下文、必要系统提示词、模型配置。
- 输出：模型生成文本、Token 使用信息、调用状态。
- 可依赖：环境变量中的 API Key、base_url、model_name、temperature、max_tokens 配置。
- 不可承担：读取全量知识库、绕过检索、直接访问 UI 状态。
- 错误面：API Key 缺失、网络失败、模型错误、响应超时、Token 信息不可得。

## citation_mapper

- 职责：将检索结果和回答依据映射为 Citation 列表。
- 输入：检索结果、Chunk、AnswerResult 生成上下文。
- 输出：Citation 列表。
- 可依赖：Chunk、Citation、DocumentBlock 回溯关系。
- 不可承担：生成模型回答、修改 Chunk 文本、伪造来源位置。
- 错误面：Chunk 缺失、来源文档名缺失、页码或等价位置不可用、quote 无法追溯。

## chat_service

- 职责：编排用户提问到回答返回的完整流程，并关联会话历史。
- 输入：session_id、用户问题、知识库状态。
- 输出：AnswerResult、会话消息记录。
- 可依赖：retrieval、llm_service、citation_mapper、Session 存储接口。
- 不可承担：文档解析、UI 控件渲染、底层索引实现。
- 错误面：会话不存在、知识库为空、检索结果不足、LLM 调用失败。

## ui_desktop

- 职责：提供桌面端界面，展示会话区、主问答区、引用区和文档导入入口。
- 输入：用户操作、应用层状态、回答结果、引用列表。
- 输出：用户可见界面状态和操作请求。
- 可依赖：应用层服务接口。
- 不可承担：业务流程决策、解析文档、建立索引、调用 LLM。
- 错误面：导入失败提示、处理状态提示、证据不足提示、调用失败提示。

## mcp_adapters

- 职责：为解析、OCR、检索、LLM 等能力提供统一接入和组织边界。
- 输入：模块能力请求、标准化参数、配置。
- 输出：标准化能力调用结果。
- 可依赖：各基础设施实现和抽象接口。
- 不可承担：通用 Agent 行为、多主体调度、插件商店、工作流平台。
- 错误面：适配器不可用、能力未配置、输入输出不符合契约。

## evaluation

- 职责：支持检索准确率、响应延迟、Token 消耗的可复现测量和证据等级判定。
- 输入：测试问题集、期望依据、系统输出、调用元数据。
- 输出：测量记录、对比结果、证据等级建议。
- 可依赖：testing_and_evaluation 的证据定义、retrieval、chat_service、llm_service 输出。
- 不可承担：设置最低分线、替代功能验收、记录每次运行流水。
- 错误面：测试集缺失、期望依据不明确、Token 信息缺失、时间记录不可用。

## 模块依赖规则

- UI 只能调用应用层服务接口。
- chat_service 负责问答流程编排。
- retrieval 只面向 Chunk。
- llm_service 隔离 OpenAI-compatible API。
- citation_mapper 只基于可回溯 Chunk 生成引用。
- module_interfaces 引用 data_contracts 中的数据对象，不重新定义字段约束。

## 禁止的模块耦合

- UI 直接调用 LLM API。
- UI 直接操作索引库。
- retrieval 直接读取原始文档全文。
- llm_service 直接读取本地知识库。
- chat_service 绕过 retrieval 直接调用模型回答知识库问题。
- evaluation 反向修改业务输出。
- mcp_adapters 扩展为通用智能体调度层。
