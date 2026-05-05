# 项目开发规格

## 项目定位

本项目是面向个人学习与科研场景的本地知识库助手。系统围绕用户本地资料提供导入、解析、清洗、切片、索引、检索、问答和来源追溯能力，帮助用户在本地资料范围内获得有依据的回答。

系统不是通用聊天机器人，不是网页系统，不是多用户协作平台，也不是通用智能体平台。MCP 在本项目中用于组织能力接入和服务边界，服务于文档处理、检索、模型调用和评价采集等能力的解耦。

## 目标用户与使用场景

目标用户是需要整理和查询本地学习、论文、课程、笔记、科研资料的个人用户。

核心使用场景：

- 用户导入本地 PDF、Markdown、TXT 文档。
- 系统将文档转换为统一中间表示。
- 系统对文本进行规则化清洗并生成可追溯 chunk。
- 系统为 chunk 生成 embedding，并写入本地 Chroma 向量数据库。
- 用户提出问题。
- 系统召回相关 chunk。
- 系统基于召回 chunk 调用 OpenAI-compatible LLM API 生成回答。
- 系统返回回答、sources 和证据充分性说明。
- 系统记录可复现的检索、耗时和 token 指标事件。

## 系统闭环

系统必须形成以下闭环：

本地文档导入 → 统一解析 → 文本清洗 → 传统 chunking → embedding 生成 → Chroma 本地向量索引 → 用户提问 → vector retrieval → 上下文组装 → OpenAI-compatible LLM 调用 → 回答生成 → sources 返回 → 指标事件记录。

实现不得跳过统一中间表示、chunk 级检索、回答来源追溯或指标采集边界。

## 功能性需求

### 文档导入

- 系统必须支持 PDF、Markdown、TXT。
- 系统必须拒绝 DOCX 与其他未列明格式，并给出明确错误。
- 系统必须为导入文档建立可追溯元数据和处理状态。

### 文档解析与中间表示

- 所有支持格式必须转换为统一 DocumentBlock 表示。
- 下游清洗、切片、索引、检索和引用都必须依赖 DocumentBlock。
- DocumentBlock 应保留文档、顺序、页码或等价位置等可追溯信息。

### 文本清洗

- 系统必须提供规则化清洗 pipeline。
- 清洗输入为 DocumentBlock，输出为仍可追溯的 DocumentBlock 或等价清洗后块。
- 清洗不得破坏标题路径、原文顺序、页码、block id 或等价位置。
- 清洗规则由 `cleaning_profile` 选择。

### chunking

- 系统使用传统 chunking 作为默认切片方式。
- chunking 输入必须来自清洗后的 DocumentBlock。
- chunking 必须保留 `source_doc_id`、`block_ids` 和位置映射。
- chunking 策略由 `chunking_strategy` 配置选择，不得写死为不可替换逻辑。
- 不同 `chunking_strategy` 产生的索引数据必须可区分。

### embedding 与向量索引

- 系统必须使用本地 `bge-base-en-v1.5` 作为默认 embedding 模型。
- embedding 通过 Python 进程内直接调用本地模型。
- embedding provider 必须支持 `embedding_device` 配置，但不得绑定特定硬件加速设备。
- 系统必须使用 Chroma 作为本地向量数据库。
- Chroma 必须使用本地持久化目录。
- 向量记录必须能追溯到 chunk、文档、清洗配置、chunking 策略和 embedding 模型。

### 检索

- 默认检索模式为 vector retrieval。
- 系统保留 keyword search fallback。
- 检索模式由 `retrieval_mode` 配置或命令参数控制。
- 检索对象是 chunk，不是整篇文档。
- 检索结果必须包含分数、来源文档、chunk id、block ids 或等价可追溯来源。

### 检索增强问答

- 问答流程必须先检索再生成。
- 默认问答路径为 vector retrieval + OpenAI-compatible LLM API。
- 当 LLM API 不可用时，允许回退到 no-LLM grounded synthesis。
- 回答必须基于 retrieved chunks，不得编造知识库外内容。
- 每次回答必须返回 answer 与 sources。
- 证据不足时必须明确说明依据不足。

### 指标采集

系统必须具备以下指标采集能力：

- Recall@k；
- 端到端响应延迟；
- embedding 耗时；
- 检索耗时；
- LLM token 消耗。

指标用于可复现记录和对比分析，不作为固定通过条件。

## 非功能性需求

- 系统必须采用分层结构，至少包含表现层、应用层、领域层、基础设施层。
- 业务流程不得堆积在 CLI、UI 或单一脚本中。
- 文档存储、解析、清洗、切片、索引建立和检索必须在本地完成。
- 外部 LLM API 只允许接收用户问题、retrieved chunks 和必要提示词。
- 禁止默认发送全量文档、全库内容或与问题无关的原始材料。
- LLM 调用必须通过独立服务接口封装。
- API key 必须通过环境变量读取，不得写入代码、配置文件或文档示例。
- 本地持久化数据必须可审计、可迁移、可重建。

## 范围外内容

- OCR 不作为实现范围；若出现扫描 PDF，仅允许通过既有工具集成边界处理，不自研 OCR 算法。
- DOCX 不作为支持格式。
- 摘要生成不作为系统能力。
- 多用户、账号、权限不作为系统能力。
- 联网知识库或云端同步不作为系统能力。
- chunking 方法对比实验不作为系统能力；系统只保留策略替换边界。
- 本地 App UI 不作为实现范围。
- 后台进程管理不作为实现范围；embedding 服务化仅作为架构演进边界。

## 成功原则

- 支持文档能完成导入、解析、清洗、切片、索引、检索和问答闭环。
- 回答始终带有可追溯 sources。
- 检索和问答路径具备 fallback 行为。
- 指标采集口径稳定，可用于多次运行的对比分析。
- 配置项、数据契约、模块接口和评价口径具备清晰 SSOT。
