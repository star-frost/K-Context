# 测试与评估规格

## 文档职责

本文档定义测试与评价口径、指标采集方式和证据等级。本文档不记录运行数据，不表达固定通过数值，不替代功能规格或数据契约。

## 功能验收对象

| 验收对象 | 证据形态 |
|---|---|
| PDF 导入 | 能生成 Document metadata 与 DocumentBlock |
| Markdown 导入 | 能生成 Document metadata 与 DocumentBlock |
| TXT 导入 | 能生成 Document metadata 与 DocumentBlock |
| 格式拒绝 | DOCX 与未列明格式给出明确错误 |
| 文本清洗 | 清洗后保留可追溯结构信息 |
| chunking | 基于清洗后 DocumentBlock 生成可追溯 Chunk |
| embedding | 本地模型能为 chunk 生成向量或给出明确不可用状态 |
| Chroma 索引 | 向量与 metadata 可持久化到本地 Chroma |
| vector retrieval | 查询可召回相关 chunk |
| keyword fallback | vector retrieval 不可用时可走关键词检索 |
| LLM answer | 回答基于 retrieved chunks 生成 |
| no-LLM fallback | LLM 不可用时返回保守 grounded synthesis 或明确依据不足 |
| sources | 回答包含可追溯来源 |
| metrics | 指标事件可写入 metrics 记录结构 |

## RAG 流程测试

知识库问答用例应覆盖：

1. 接收用户问题。
2. 构造检索查询。
3. 召回 Top-K chunk。
4. 组装上下文。
5. 调用 LLM client 或进入 no-LLM fallback。
6. 生成回答。
7. 映射 sources。
8. 记录指标事件。

若检索或 sources 缺失，该用例不满足 RAG 流程要求。

## Recall@k

Recall@k 用于衡量 Top-K 检索结果是否覆盖人工标注的相关依据。

基本口径：

- 对每个测试问题，人工标注相关文档或相关依据片段。
- 系统返回 Top-K 检索结果。
- 若 Top-K 中包含标注相关依据，则该问题视为被召回。
- 汇总多个问题的召回覆盖情况。

标注要求：

- ground truth 必须基于人工标注。
- 不得使用模型自评替代人工标注。
- 支持 doc-level 标注。
- 支持 evidence/chunk-level 标注。
- chunk id 不稳定时，应使用 document、source span、evidence text 或 evidence text hash 维持可比性。

Recall@k 只作为评价指标，不作为固定通过条件。

## 端到端响应延迟

端到端响应延迟从用户提交问题开始，到 answer 与 sources 可返回为止。

应记录：

- operation；
- start timestamp；
- end timestamp；
- duration_ms；
- retrieval_mode；
- top_k；
- 是否使用 LLM；
- 是否使用 fallback。

该指标用于比较不同运行配置下的耗时差异，不表达固定通过数值。

## embedding 耗时

embedding 耗时用于记录 chunk embedding 生成过程。

应记录：

- embedding model；
- embedding device；
- chunk count；
- batch size，如实现可获得；
- duration_ms；
- success 或 failure 状态；
- failure reason，如存在。

不得在规范中写死特定硬件加速设备或性能数据。

## 检索耗时

检索耗时用于记录一次查询从接收 query 到得到 Top-K 结果的耗时。

应记录：

- retrieval_mode；
- top_k；
- vector_store_type；
- 是否生成 query embedding；
- duration_ms；
- result count；
- fallback 状态。

vector retrieval 与 keyword fallback 的记录应可区分。

## LLM token 消耗

TokenUsage 来源只允许：

| 来源 | 含义 |
|---|---|
| `api_usage` | OpenAI-compatible API 响应提供 usage |
| `estimated` | API 未返回 usage，由本地估算 |
| `unavailable` | 无法获取且未估算 |

优先使用 API 返回 usage。无法获得时可以估算。无法估算时必须标记为 `unavailable`。

应记录：

- prompt tokens；
- completion tokens；
- total tokens；
- source；
- llm model；
- 是否使用 fallback。

不得记录 API key。

## metrics 记录口径

指标事件建议写入 `.kcontext/metrics.jsonl` 或等价结构。

`sessions.jsonl` 与 `metrics.jsonl` 的边界：

| 文件 | 记录内容 |
|---|---|
| `sessions.jsonl` | 会话、问题、回答、sources、交互历史 |
| `metrics.jsonl` | 耗时、token、检索、embedding、评价等测量事件 |

指标记录不得替代会话历史；会话历史不得混写为指标事件。

## fallback 测试

应覆盖以下不可用状态：

- embedding 模型不可用；
- Chroma 不可用；
- LLM API 不可用；
- vector index 缺失；
- retrieved chunks 为空。

预期行为：

- 系统给出明确状态。
- 允许的 fallback 路径可运行。
- 输出中不伪装默认路径成功。
- 指标事件能记录实际路径。

## 证据不足测试

证据不足测试应确认：

- 检索为空时，系统明确说明依据不足。
- 召回内容不能支撑结论时，系统不伪造确定性回答。
- sources 为空时，回答正文必须体现依据不足。
- no-LLM fallback 只基于 retrieved chunks 组织回答。

## 测试集要求

测试集应包含：

- 覆盖 PDF、Markdown、TXT 的样例文档。
- 与文档内容相关的问题。
- 每个问题对应的人工标注依据。
- 至少一类依据不足问题。
- 能体现结构边界的材料，例如标题、列表、代码块、跨页内容或纯文本段落。

测试集不得要求系统对知识库外内容伪造答案。

## 对比评估方法

对比评估应固定：

- 测试文档集合；
- 测试问题集合；
- 人工标注依据；
- 指标采集口径；
- 配置记录方式。

多次运行可比较 Recall@k、端到端响应延迟、embedding 耗时、检索耗时和 token 消耗的变化。文档只定义比较方法，不保存运行数据。

## 证据等级

系统评价只使用以下三种证据等级：

| 等级 | 含义 | 判定语义 |
|---|---|---|
| 证据不足 | 证据链缺口明显，无法支撑关键结论 | 只能说明执行过验证动作，不能说明结论成立 |
| 基本充分 | 覆盖核心路径并能支撑主要结论，但边缘或异常覆盖有限 | 可支持目标基本成立，保留已知不确定性 |
| 充分 | 关键路径、关键约束与主要异常面均有一致证据支撑 | 可支持结论稳健成立，剩余风险可陈述且可管理 |

## 禁止的评价写法

- 不把某个数值作为通过条件。
- 不记录每次运行流水。
- 不把研究性指标替代功能验收。
- 不用模型自评替代人工标注。
- 不写性能数据。
