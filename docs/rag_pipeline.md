# RAG 管线规格

## 文档职责

本文档描述系统的 RAG 管线顺序、策略配置如何影响流程、索引与查询路径、fallback 关系和可替换边界。字段结构以 `data_contracts.md` 为准，模块输入输出以 `module_interfaces.md` 为准，评价口径以 `testing_and_evaluation.md` 为准。

## 管线总览

系统 RAG 管线为：

clean → chunk → embed → vector store → vector retrieval → LLM answer

该管线建立在文档导入和统一解析之后。DocumentBlock 是清洗、切片、索引、检索和来源追溯的统一输入基础。

## 索引管线

`kb index` 是索引构建主入口，职责顺序为：

1. 读取 DocumentBlock。
2. 按 `cleaning_profile` 执行文本清洗。
3. 按 `chunking_strategy` 执行传统 chunking。
4. 写入或更新 `chunks.jsonl`。
5. 按 `embedding_model` 与 `embedding_device` 生成 embedding。
6. 将向量和 metadata upsert 到 Chroma。
7. 记录 index、embedding、向量写入相关指标事件。

`kb add` 不隐式替代 `kb index`。导入文档后，显式索引构建仍是向量检索可用的前置动作。

## 清洗策略

`cleaning_profile` 用于选择清洗规则集合。清洗可以处理空白、不可见字符、重复空行、无效文本块和格式噪声。清洗不得破坏：

- document id；
- block id；
- block 顺序；
- 标题路径；
- 页码或等价位置；
- 可用于引用的文本片段。

清洗后无有效文本时，应产生明确状态，不得生成不可追溯 chunk。

## chunking 策略

默认 chunking 为传统结构感知切片。`chunking_strategy` 用于选择具体切片策略。

要求：

- chunking 输入为清洗后的 DocumentBlock。
- chunk 必须保留 source document 与 block ids。
- chunk id 可以随策略变化而变化，因此评估标注不得只依赖单次生成的 chunk id。
- 不同 `chunking_strategy` 的向量记录必须可区分。
- 策略替换不得改变检索、问答和 sources 的上层接口。

## embedding 管线

默认 embedding 模型为本地 `bge-base-en-v1.5`。embedding provider 通过 Python 进程内直接调用本地模型。

输入：chunk text 与 embedding 配置。  
输出：向量、维度、模型标识、文本 hash 与创建时间。

embedding 生成必须记录可用于缓存判断和重建判断的信息。模型不可用、device 不可用或向量维度不匹配时，不得写入错误向量。

## Chroma 写入管线

Chroma 是默认 vector store。向量写入必须携带 metadata，用于来源追溯、配置区分和索引重建判断。

collection metadata 应能区分：

- embedding model；
- embedding dim；
- chunking strategy；
- cleaning profile；
- index version。

vector record metadata 应能追溯到 chunk、document、block、清洗配置、切片策略和 embedding 配置。

## 查询管线

`kb search` 默认执行 vector retrieval：

1. 读取查询文本。
2. 根据 `retrieval_mode` 选择检索模式。
3. vector 模式下生成 query embedding。
4. 查询 Chroma。
5. 返回 top_k 检索结果。
6. 每条结果包含 score、chunk id、source document、block ids 和文本片段。

当 vector retrieval 不可用时，可使用 keyword fallback。fallback 结果必须标明实际 retrieval mode。

## 问答管线

`kb ask` 默认路径为：

1. 接收 question。
2. 调用 retrieval 服务召回 chunks。
3. 组装上下文。
4. 调用 OpenAI-compatible LLM API。
5. 映射 sources。
6. 返回 answer、evidence_level、sources。
7. 记录端到端延迟、检索耗时、token usage 等指标事件。

问答不得绕过 retrieval 直接调用模型回答知识库问题。LLM 不可用时，允许 no-LLM grounded synthesis fallback。fallback 回答必须保守，且不得编造 retrieved chunks 之外的信息。

## retrieval mode

`retrieval_mode` 控制检索路径：

- `vector`：默认向量检索；
- `keyword`：关键词检索 fallback；
- `hybrid`：保留为可组合模式，具体字段和接口仍以数据契约与模块接口为准。

若配置或参数请求的模式不可用，系统必须返回明确状态或使用允许的 fallback，并在输出或指标事件中记录实际路径。

## top_k

`top_k` 控制召回数量。字段语义和默认值以 `data_contracts.md` 为准。Recall@k 的 k 值可引用该配置或评价任务指定值，但评价文档不得重定义默认值。

## fallback 规则

| 不可用能力 | 允许 fallback | 要求 |
|---|---|---|
| embedding 模型 | keyword retrieval | 明确 vector retrieval 不可用 |
| Chroma | keyword retrieval | 不得返回伪造向量分数 |
| LLM API | no-LLM grounded synthesis | 回答必须基于 retrieved chunks |
| retrieved chunks 为空 | evidence insufficient response | 不得编造答案 |

## 可替换边界

系统允许通过配置替换清洗规则、chunking 策略、检索模式、embedding 模型和 vector store 实现。替换必须保持：

- DocumentBlock 到 Chunk 的可追溯链；
- Chunk 到 VectorRecord 的可追溯链；
- SearchResult 到 Source 的可追溯链；
- 指标事件的统一记录格式；
- AnswerResult 的输出结构。

chunking 方法对比不属于本文档描述的实现目标；本文档只规定可替换与可追溯边界。
