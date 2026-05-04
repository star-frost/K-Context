# 数据契约

## Document 元数据

Document 表示导入到系统中的本地文件记录。

| 字段 | 含义 | 约束 |
|---|---|---|
| doc_id | 文档唯一标识 | 全局唯一，创建后稳定 |
| file_name | 原始文件名 | 用于展示和引用来源 |
| file_type | 文件类型 | 仅允许 pdf、md、txt 三类归一化值 |
| storage_ref | 本地路径或内部存储标识 | 不得指向云端同步地址作为默认存储 |
| imported_at | 导入时间 | 用于排序和追踪 |
| status | 处理状态 | 仅允许已导入、已解析、已切块、已建立索引、处理失败 |
| error_message | 失败说明 | 处理失败时应提供可读说明 |

## DocumentBlock IR

DocumentBlock 是统一 IR 的最小块结构。所有格式解析结果都必须转换为该结构后再进入清洗、切块、索引、检索和引用流程。

| 字段 | 含义 | 约束 |
|---|---|---|
| block_id | 块唯一标识 | 全局唯一或在文档内可唯一定位 |
| source_doc_id | 来源文档标识 | 必须对应 Document.doc_id |
| source_doc_name | 来源文档名 | 用于引用展示 |
| page | 页码 | 非分页文档可为空 |
| order | 块在原文中的顺序 | 必须可用于恢复原文顺序 |
| block_type | 块类型 | 只允许 title、paragraph、list、table、code、figure、unknown |
| heading_path | 标题路径 | 无标题时为空列表 |
| text | 文本内容 | 清洗后不得为空 |
| bbox | 坐标信息 | 无坐标时可为空，字段必须存在 |

DocumentBlock 不得绕过清洗规则直接进入索引；不得以不同格式维护互不兼容的后处理结构。

## Chunk

Chunk 是检索索引的基本对象，由一个或多个 DocumentBlock 生成。

| 字段 | 含义 | 约束 |
|---|---|---|
| chunk_id | 切块唯一标识 | 必须可被 Citation 引用 |
| source_doc_id | 来源文档标识 | 必须对应 Document.doc_id |
| source_doc_name | 来源文档名 | 用于引用展示 |
| page_start | 起始页 | 非分页文档可为空 |
| page_end | 结束页 | 非分页文档可为空 |
| heading_path | 标题路径 | 继承或合并来源块标题路径 |
| block_ids | 来源块标识列表 | 必须能回溯到 DocumentBlock |
| text | Chunk 文本 | 目标长度为 200 至 1000 字符，最终不得超过 1500 字符 |

Chunk 必须基于 IR 块序列生成，不得直接基于原始全文生成。

## Citation

Citation 表示回答依据。

| 字段 | 含义 | 约束 |
|---|---|---|
| chunk_id | 引用 Chunk 标识 | 必须对应可检索 Chunk |
| source_doc_name | 来源文档名 | 必须可展示给用户 |
| page_start | 起始页 | 非分页文档可为空 |
| page_end | 结束页 | 非分页文档可为空 |
| quote | 引用文本片段 | 应来自 Chunk 文本或其可追溯片段 |

Citation 必须能回溯到 Chunk，再回溯到 DocumentBlock 和来源文档位置。

## AnswerResult

AnswerResult 表示一次系统回答的最小返回对象。

| 字段 | 含义 | 约束 |
|---|---|---|
| answer_text | 回答正文 | 证据不足时必须明确说明依据不足 |
| citations | 引用列表 | 可以为空，但为空时回答必须说明证据不足 |

系统禁止只返回字符串形式的回答。

## Session 与 Message

Session 用于管理单个用户的多会话问答历史，不用于用户隔离。

| 对象 | 字段 | 含义 |
|---|---|---|
| Session | session_id | 会话唯一标识 |
| Session | created_at | 创建时间 |
| Session | messages | 消息历史 |
| Session | answer_refs | 关联回答记录 |
| Message | message_id | 消息唯一标识 |
| Message | role | user 或 assistant |
| Message | content | 消息正文 |
| Message | created_at | 消息时间 |
| Message | answer_result_id | 助手消息对应回答记录，可为空 |

## 字段约束

- `doc_id`、`block_id`、`chunk_id`、`session_id`、`message_id` 必须稳定且可追踪。
- `block_type` 不得扩展到未声明取值。
- `status` 不得使用不透明状态。
- `page`、`bbox` 可以为空，但字段必须存在。
- `heading_path` 必须保留结构信息，不得在清洗中丢失。
- top_k 默认值为 5，由检索模块使用。

## 回溯关系

引用回溯链必须成立：

Citation → Chunk → DocumentBlock → Document → 原始本地文件位置。

任何回答中的引用都应能显示文档名、位置标识和引用文本片段。若无法建立该链路，则不得将该片段作为可靠引用展示。

## 禁止绕过的数据路径

- 原始文档文本不得绕过 IR 直接进入切块。
- 原始文档全文不得默认直接进入提示词。
- 检索不得面向整篇文档替代 Chunk。
- 回答不得绕过 Citation 直接显示无来源结论。
- OCR 输出不得形成与 IR 并行的后处理结构。
