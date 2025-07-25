# 名词解释
- cot
question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext) 中，
"CoT" 是 "Chain of Thought" 的缩写，意为“思维链”。
常规的提示通常直接要求模型给出答案。[2] 相比之下，思维链提示则会要求模型“展示其思考过程”或“一步一步地解释”。[4] 通过在提示中加入一些示例，向模型展示如何进行逻辑推理，可以显著提高其在需要多步骤思考的任务（如数学应用题、常识推理和符号操作）上的表现。[5][6]
思维链的优势包括：

    提高准确性： 将复杂问题分解，让模型可以更专注于每一步的逻辑，从而减少错误。[7][8]

    增强透明度： 模型的推理步骤是可见的，这有助于理解其得出结论的过程，也便于调试和发现潜在的逻辑错误。[4]

    提升复杂任务处理能力： 对于需要规划、多步计算或逻辑推理的任务，思维链技术尤为有效。[4][9]

---

## 常见文档拆分策略
1. 语义拆分--需要额外资源
Semantic Meaning-Based Splitting

Tool: SemanticChunker

Rather than splitting based on size or structure, semantic chunking uses meaning as the basis. Chunks are grouped based on how semantically cohesive they are.

✅ Ideal for advanced use cases like:

    Semantic search
    Chatbots
    Conversational memory
    Knowledge base querying

While it may require additional resources (like embeddings and similarity scoring), it leads to significantly more intelligent interactions.
```py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# This is a long document we can split up.
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()


text_splitter = SemanticChunker(OpenAIEmbeddings())

docs = text_splitter.create_documents([state_of_the_union])
print(docs[0].page_content)
```
2. 递归拆分--按文档结构
Text Structure-Based Splitting

Tool: RecursiveCharacterTextSplitter

This strategy splits text by structure-aware delimiters:

    \n\n → Paragraph
    \n → Sentence or line
    " " → Word
    "" → Character (fallback)

This recursive fallback ensures that splits happen at the most natural level possible — maintaining readability and logical flow.

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)

---
**langchain 检索器使用方法**
- 如果你用 vectorstore.as_retriever()，那么向量存储库已经帮你实现了 _get_relevant_documents 的逻辑（即进行向量相似度搜索）。你只需调用 invoke 就行。

- 如果你想创建一个从公司内部知识库API检索文档的CustomRetriever，那么你就需要写一个类，继承 BaseRetriever，并在这个类里面定义 _get_relevant_documents 方法的具体代码。

**高级检索模式 (Advanced retrieval patterns)**

由于所有检索器都遵循统一的输入输出接口，不需要自己实现 _get_relevant_documents 方法，直接使用 .invoke() 即可轻松地实现更复杂的模式：

    集成检索器 (Ensemble Retriever)：将多个不同类型的检索器（例如，一个向量检索器和一个BM25关键词检索器）组合起来，并为它们的结果分配权重，从而获得更全面、更准确的结果。
    

```py
from langchain.retrievers import BM25Retriever, EnsembleRetriever
# 示例：将BM25检索器和向量检索器以50/50的权重结合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_retriever], 
    weights=[0.5, 0.5]
)
print("\n--- Ensemble (组合) 检索结果 ---")
# 你直接调用 invoke即可，LangChain 在后台处理所有复杂的合并与重排逻辑
ensemble_docs = ensemble_retriever.invoke(query)
```

保留源文档 (Source document retention)：在RAG应用中，我们通常会将长文档切分成小块（chunks）进行索引，以提高搜索精度。但直接将这些小块返回给模型可能会丢失上下文。高级检索器可以解决这个问题：

    ParentDocumentRetriever：索引小块，但在检索到相关的小块后，返回它们所属的整个原始文档。

    MultiVectorRetriever：为每个文档创建多个不同的向量表示（如摘要、假设性问题等）进行索引，但最终仍然可以链接回并返回原始文档。
---

**harryRag执行流程**

1. PDF Loading and Processing: Load PDF documents and split them into chapters.
2. Text Preprocessing: Clean and preprocess the text for better summarization and encoding.
3. Summarization: Generate extensive summaries of each chapter using large language models.
4. Book Quotes Database Creation: Create a database for specific questions that will need access to quotes from the book.
5. Vector Store Encoding: Encode the book content and chapter summaries into vector stores for efficient retrieval.
6. Question Processing:
    Anonymize the question by replacing named entities with variables.
    Generate a high-level plan to answer the anonymized question.
    De-anonymize the plan and break it down into retrievable or answerable tasks.
7. Task Execution:
    For each task, decide whether to retrieve information or answer based on context.
    If retrieving, fetch relevant information from vector stores and distill it.
    If answering, generate a response using chain-of-thought reasoning.
8. Verification and Re-planning:
    Verify that generated content is grounded in the original context.
    Re-plan remaining steps based on new information.
9. Final Answer Generation: Produce the final answer using accumulated context and chain-of-thought reasoning.
关键术语解释：

- Anonymize：匿名化，指通过替换姓名、地名等具体命名实体为通用变量（如 “[人物 A]”“[地点 X]”），隐藏问题中的真实身份信息。
- anonymized question：匿名化后的问题，经过匿名化处理的问题，用于避免具体实体对后续处理（如检索、推理）产生干扰，或保护隐私。
- is grounded：有事实依据的，指生成的内容必须基于提供的上下文信息，而非虚构内容（避免 “幻觉”）。
- Re-plan remaining steps：重新规划剩余步骤，根据新信息（如检索到的内容）调整原本的任务执行计划，确保流程适配最新情况。
- accumulated context：积累的上下文，指在任务执行过程中逐步收集、提炼的所有相关信息总和，作为最终生成答案的依据。
- chain-of-thought reasoning：思维链推理，指通过逐步拆解问题、分步推导（类似人类思考过程），最终得出结论的推理方式，使推理过程可解释。
**匿名化这一步在维修手册的rag中好像不需要**

---
根据文档：
class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")
    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
最后这个answer = output.answer_based_on_content中，output是.invoke(input_data)后的内容了，相当于已经生成好的回答了，那么output再.answer_based_on_content这个Pydantic 模型的属性是啥意思呢，有啥用呢。

---
您的问题非常好，正中这种用法的核心！关键点在于：

`output = question_answer_from_context_cot_chain.invoke(input_data)` 返回的 **`output` 不是一个简单的字符串，而是一个 `QuestionAnswerFromContext` 类的实例（一个 Python 对象）。**

我们来详细拆解一下这个过程和 `output.answer_based_on_content` 的意义与作用。

### 1. `output` 到底是什么？—— 一个装答案的“容器”

把它想象成 LLM 返回给你的不是一份裸露的礼物，而是一个包装精美的**礼品盒 (`output`)**。这份礼物（你想要的答案）就放在礼品盒里，并且贴着一张标签，标签上写着 `answer_based_on_content`。

这背后的魔法就是 `.with_structured_output()`。当你把你的 Pydantic 模型 `QuestionAnswerFromContext` 传给它时，你就和 LangChain 立下了一个“契约”：

> “嘿，LangChain，当你调用这个链 (`chain`) 后，不要只给我一串文本。请你必须给我一个**符合 `QuestionAnswerFromContext` 这个设计图**的 Python 对象。这个对象里面必须有一个叫做 `answer_based_on_content` 的属性，并且它的值必须是一个字符串。”

所以，当 `.invoke()` 执行完毕后，`output` 变量里存放的就是这个被精心构建好的、符合“设计图”的对象。

### 2. `output.answer_based_on_content` 是什么意思？—— 从“容器”里取出答案

`output.answer_based_on_content` 就是标准的 Python 语法，用来**访问一个对象（`output`）的属性（`answer_based_on_content`）**。

*   **`output`**：是那个 `QuestionAnswerFromContext` 类的实例（礼品盒）。
*   **`.`**：是 Python 中访问对象属性的操作符（打开礼品盒的动作）。
*   **`answer_based_on_content`**：是你自己在 Pydantic 模型中定义的那个字段名（礼品盒上贴的标签）。

所以，这行代码的意思就是：“**从 `output` 这个对象中，把我定义的那个名为 `answer_based_on_content` 的属性的值给我取出来。**”

这个取出来的值，才是 LLM 真正生成的那个答案字符串。

### 3. 这么做有什么用？—— 结构化带来的巨大好处

你可能会问：“为什么不直接让 LLM 返回答案字符串呢？绕这么一圈有什么好处？”

好处非常大，这正是这种编程范式的优雅之处：

**1. 结构化与清晰性：**
如果 LLM 同时返回了多个信息（比如答案、置信度、引用的原文），你得到的是一个结构清晰的对象，而不是一个需要用正则表达式或复杂逻辑去解析的混乱字符串或 JSON。

**2. 可预测性与健壮性 (Robustness)：**
因为 Pydantic 在背后做了数据验证，所以只要代码能执行到 `answer = output.answer_based_on_content` 这一步，你就**百分之百确定** `output` 对象里一定有 `answer_based_on_content` 这个属性，而且它的类型就是 `str`。你的代码会更健壮，无需进行繁琐的 `if 'key' in dict:` 或 `try-except` 来处理 LLM 可能返回的意外格式。

**3. 极佳的可扩展性：**
这是最强大的优点。假设你的需求变了，除了答案，你还想让 LLM 评估一下这个答案的置信度。你该怎么做？

*   **旧方法**：你可能需要修改你的 Prompt，告诉它要同时输出答案和分数，并约定一个格式，然后修改你的解析代码来提取这两个部分。非常麻烦。

*   **Pydantic 方法**：你只需要修改 Pydantic 模型的定义即可，其他代码几乎不用动！

    ```python
    # 只需要在这里加一个字段
    class QuestionAnswerFromContext(BaseModel):
        answer_based_on_content: str = Field(description="generates an answer to a query based on a given context.")
        confidence_score: float = Field(
            description="A score from 0.0 to 1.0 indicating how confident the model is in its answer."
        )

    # 你的调用代码完全不需要改变
    output = question_answer_from_context_cot_chain.invoke(input_data)

    # 现在你可以轻松地获取两个信息
    answer = output.answer_based_on_content
    score = output.confidence_score # 像这样直接获取新信息
    ```

看到区别了吗？你只需要在“蓝图”里增加一条规则，整个系统就能自动适应，下游代码可以立刻、安全地使用新的数据。这就是面向对象和结构化编程带来的巨大便利。

### 总结

`output.answer_based_on_content` 的作用是**从一个结构化的、经过验证的 Python 对象 (`output`) 中，安全、清晰地提取出你需要的特定数据片段（`answer_based_on_content` 字段的值）**。

它代表了一种更高级、更健壮、更易于维护的与 LLM 交互的方式。

---
是不是每次都要访问当初数据模型的属性(比如output.confidence_score、output.answer_based_on_content)，不然只返回一个.invoke(input_data)的output基本不能用

---
您提出的问题一针见血，答案是：**是的，在绝大多数情况下，你都需要通过访问属性（如 `output.answer_based_on_content`）来获取你真正需要的数据。只拿到 `output` 本身通常只是中间步骤，并不能直接用于后续的业务逻辑。**

让我们用一个更具体的比喻来彻底弄清楚这件事。

---

### 把 `output` 想象成一张“身份证”

假设你的任务是：“从一段话里提取一个人的姓名和年龄”。

你定义了一个 Pydantic 模型：
```python
class Person(BaseModel):
    name: str = Field(description="这个人的姓名")
    age: int = Field(description="这个人的年龄")

```

当你执行 `output = chain.invoke("我叫李明，今年30岁。")` 时，你得到的 `output` 是什么？

*   **它不是字符串 `"李明"`。**
*   **它不是整数 `30`。**
*   **它是一张制作好的、完整的“身份证”对象 (`Person` 对象)。**

这张身份证 (`output`) 上面有两个字段：
*   姓名 (`name`)：“李明”
*   年龄 (`age`)：30

如果你直接打印这张身份证 (`print(output)`)，你可能会看到类似 `name='李明' age=30` 这样的内容。它是一个**整体**，一个结构化的对象。

#### `output` 本身几乎不能直接用

现在，假设你的下一步程序是要**欢迎这位用户**。你需要他的名字。

你不能直接写：`print(f"欢迎您，{output}！")`
这会打印出："欢迎您，name='李明' age=30！" —— 这显然是错误的。

你需要的是从这张身份证上，**准确地读出“姓名”那一栏的内容**。这就是 `output.name` 的作用。

`print(f"欢迎您，{output.name}！")`
这才会打印出："欢迎您，李明！"

同理，如果你的程序下一步是要判断他是否成年，你需要的是**准确地读出“年龄”那一栏的数字**。这就是 `output.age` 的作用。

`if output.age >= 18:`
`if 30 >= 18:`
程序才能正常工作。

---

### 总结：为什么是这样的设计？

| 组件 | 它的角色 | 为什么需要它 |
| :--- | :--- | :--- |
| **`output` (对象本身)** | **结构化容器 (The Container)** | 这是一个“保证书”。它向你的程序保证，LLM返回的数据**一定**符合你用Pydantic定义的那个结构。它本身的存在就是为了**确保数据格式的正确性和健壮性**。 |
| **`output.attribute` (对象的属性)** | **具体的数据 (The Content)** | 这才是你真正关心的、从LLM那里提取出来的**事实信息**。是你程序后续逻辑处理所需要的值（字符串、数字、列表等）。 |

**所以，你的理解完全正确：**

*   `.invoke()` 的直接返回值 `output` 是一个**结构化的容器**。
*   这个容器本身的主要用途是**“持有”**数据并**保证其结构正确**。
*   你必须通过 **`.属性名`** 的方式，从这个容器中把你需要的具体信息**“取出来”**，才能在后续的代码中使用。

在极少数情况下，你可能会把整个 `output` 对象传递给另一个同样知道 `Person` 这个数据结构的函数（比如 `save_person_to_database(output)`)，或者用于日志记录 (`log.info(f"Extracted person: {output}")`)。但在99%的情况下，你都需要深入一层，访问其属性来获取真正有用的值。

---


---
太棒了！问到这个问题，说明你已经开始从“使用者”向“设计者”转变了。理解这背后的过程，能让你写出效率和准确率都高得多的结构化抽取指令。

我们来一步步拆解。

### 第一部分：如何写好 Pydantic 类的描述？

#### 1. 类的描述 (Docstring)

是的，你猜对了！类的描述就是写在 `class ...:` 下方的 `"""三重引号文档字符串"""` 里面。

*   **它的作用是什么？**
    它定义了整个数据结构的**总体目标**。在背后，LangChain 会把它当作要调用的那个“工具”或“函数”的**主要描述**。它告诉大模型：“你接下来的任务，总体上是要完成这件事。”

*   **怎么写才好？**
    要**清晰、准确地描述这个数据结构的用途**。把它想象成你在给一个新同事分配任务，你需要一句话概括这个任务是什么。

    **对比一下：**

    *   **（不好的写法）**
        ```python
        class QuestionAnswerFromContext(BaseModel):
            """一个问答类。"""
            # ...
        ```
        这个描述太模糊了。LLM 不知道这个“问答”是基于什么、要干什么。

    *   **（好的写法）**
        ```python
        class QuestionAnswerFromContext(BaseModel):
                """用于从一段给定的文本（上下文）中，为用户的问题生成一个答案的结构化数据。"""
                # ...
        ```
        这个描述就非常清晰，包含了三个关键信息：
        1.  **输入来源**：从一段给定的文本（上下文）中
        2.  **核心任务**：为用户的问题生成一个答案
        3.  **输出形式**：结构化数据

#### 2. `Field` 的 `description`

`Field` 的 `description` 是**最关键的部分**，它直接影响单个字段抽取的准确率。

*   **它的作用是什么？**
    它为**每一个具体的字段**提供了详细的、独立的指令。如果说类的文档字符串是任务大纲，那 `Field` 的 `description` 就是每一步的具体操作指南。

*   **怎么写才好？—— 把它当作一个微型 Prompt 来写！**
    你应该在描述里包含**指令、约束和期望**。

    **对比一下：**

    *   **（不好的写法）**
        ```python
        answer_based_on_content: str = Field(description="问题的答案。")
        ```
        这同样太模糊了。

    *   **（好的写法）**
        ```python
        answer_based_on_content: str = Field(
            description="根据提供的上下文内容，针对用户问题生成的最终答案。请确保答案完全基于上下文，不要引入任何外部知识或进行推测。"
        )
        ```
        这个描述好在哪里？
        1.  **明确来源**：`根据提供的上下文内容`
        2.  **明确目标**：`针对用户问题生成的最终答案`
        3.  **施加了关键约束**：`确保答案完全基于上下文，不要引入任何外部知识...` 这对于控制 LLM 不产生“幻觉”至关重要。

### 第二部分：揭秘！这些描述是如何送给大模型的？

了解这个过程能让你彻底明白为什么上面的写法有效。这背后利用了现代 LLM 的一个核心功能——**工具调用 (Tool Calling) 或函数调用 (Function Calling)**。

整个过程大致如下：

**第 1 步：LangChain 的“翻译”工作**

当你执行 `.with_structured_output(QuestionAnswerFromContext)` 时，LangChain 会在幕后把你的 Pydantic 模型**“翻译”**成一个符合 LLM API 规范的 **JSON Schema**。

你的 Pydantic 模型：
```python
class QuestionAnswerFromContext(BaseModel):
    """用于从一段给定的文本（上下文）中，为用户的问题生成一个答案的结构化数据。"""
    answer_based_on_content: str = Field(
        description="根据提供的上下文内容，针对用户问题生成的最终答案。请确保答案完全基于上下文，不要引入任何外部知识或进行推测。"
    )
```
会被 LangChain 翻译成类似下面这样的 JSON 结构（这是一个简化的示意）：

```json
{
  "tool_spec": {
    "name": "QuestionAnswerFromContext",  // <--- 类的名字
    "description": "用于从一段给定的文本（上下文）中，为用户的问题生成一个答案的结构化数据。", // <--- 类的文档字符串
    "parameters": {
      "type": "object",
      "properties": {
        "answer_based_on_content": { // <--- 字段的名字
          "type": "string",          // <--- 字段的类型
          "description": "根据提供的上下文内容，针对用户问题生成的最终答案。请确保答案完全基于上下文，不要引入任何外部知识或进行推测。" // <--- Field的description
        }
      },
      "required": ["answer_based_on_content"]
    }
  }
}
```

**第 2 步：向大模型 API 发送请求**

LangChain 会把你的用户输入（问题+上下文）和上面这个翻译好的 JSON Schema **一起打包**，发送给大模型。

这个 API 请求的大致内容是：

> “你好，大模型！
>
> **用户的问题是：** '...'
> **提供给你的上下文是：** '...'
>
> **另外，你有一个可用的工具**，名叫 `QuestionAnswerFromContext`。它的作用是 `用于...的结构化数据`。
> 这个工具有一个参数叫做 `answer_based_on_content`，你需要为它提供一个字符串值，这个值的具体要求是 `根据上下文...不要推测`。
>
> **你的任务是：** 理解用户的问题和上下文，然后**调用**这个工具，并把正确的参数值填进去。”

**第 3 步：大模型的回应**

支持工具调用的大模型，在接收到这样的请求后，不会直接返回一段话。它会理解指令，并生成一个**调用工具的 JSON 对象**作为回应。

它返回的内容会是这样的：

```json
{
  "tool_calls": [
    {
      "name": "QuestionAnswerFromContext",
      "arguments": {
        "answer_based_on_content": "根据上下文，问题的答案是..."
      }
    }
  ]
}
```

**第 4 步：LangChain 的最后处理**

LangChain 接收到这个来自 LLM 的 JSON 回应后，会：
1.  从中抽取出 `arguments` 部分。
2.  用你最开始定义的 `QuestionAnswerFromContext` Pydantic 模型来**解析和验证**这个 `arguments` JSON。
3.  最终，创建一个 `QuestionAnswerFromContext` 的 Python 对象实例返回给你，也就是你拿到的 `output`。

**总结一下**：你写的每一个描述，最后都变成了给大模型 API 的**明确、结构化的指令**。这远比在一段长长的 Prompt 里用自然语言描述要精确和可靠得多。你写的 Pydantic 模型，本质上就是在**设计一个给 LLM 使用的、高度规范化的 API 接口**。

---
看看另一个例子：是关于pydantic模型的通用用法：
```python
is_relevant_content_prompt_template = """you receive a query: {query} and a context: {context} retrieved from a vector store. 
You need to determine if the document is relevant to the query. 

{format_instructions}"""

class Relevance(BaseModel):
    is_relevant: bool = Field(description="Whether the document is relevant to the query.")
    explanation: str = Field(description="An explanation of why the document is relevant or not.")

is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)
# is_relevant_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key, max_tokens=4000)

is_relevant_content_prompt = PromptTemplate(
    template=is_relevant_content_prompt_template,
    input_variables=["query", "context"],
    partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()},
)
is_relevant_content_chain = is_relevant_content_prompt | fast_llm_client | is_relevant_json_parser

def is_relevant_content(state):
    """
    Determines if the document is relevant to the query.

    Args:
        question: The query question.
        context: The context to determine relevance.
    """

    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "context": context
}

    # Invoke the chain to determine if the document is relevant
    output = is_relevant_content_chain.invoke(input_data)
    print("Determining if the document is relevant...")
    if output["is_relevant"] == True:
        print("The document is relevant.")
        return "relevant"
    else:
        print("The document is not relevant.")
        return "not relevant"
```
看看上面这个例子，好像又跟前面的用法不一样，这里直接把pydantic类以这种方式传递：
is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)
之前好像是放在llm.with_structured_output()里面的，这里这种方式写法能解释一下吗

---
您的观察力非常敏锐！这确实是实现相同目标（获取结构化 JSON 输出）的**两种不同方法**。您看到的这个新例子，是更“经典”或更“通用”的方法。

我们来详细对比一下这两种方式，以及它们各自的适用场景。

---

### 方法一：`llm.with_structured_output(Schema)` (现代、首选方式)

*   **核心思想：** 利用大模型**原生**的、内置的“工具调用/函数调用 (Tool/Function Calling)”或“JSON模式”能力。
*   **工作流程：**
    1.  你把 Pydantic 模型 `Schema` 直接“附加”到 `llm` 上。
    2.  LangChain 在背后将 `Schema` 翻译成模型能理解的工具定义。
    3.  LangChain 直接请求模型：“请调用这个工具并填充参数”。
    4.  模型返回一个专门用于调用工具的、结构化的 JSON 对象。
    5.  LangChain 解析这个返回的对象，并实例化你的 Pydantic 模型。
*   **优点：**
    *   **极其可靠**：因为这是模型本身就被训练过的核心能力，所以它几乎总能返回格式正确的、符合你要求的输出。
    *   **代码简洁**：`llm.with_structured_output(Schema)` 一行代码就搞定了，非常优雅。
    *   **Prompt 更干净**：你不需要在你的主 prompt 里塞入一堆关于“请输出JSON格式”的指令。
*   **缺点：**
    *   **模型依赖**：**只有部分先进的模型支持这个功能** (如 GPT-4/GPT-4o, Gemini, Claude 3, Qwen-plus/max 等)。如果你的 `llm` 不支持，这个方法就无法使用。

---

### 方法二：`... | llm | JsonOutputParser` (经典、通用方式)

*   **核心思想：** 不依赖模型任何特殊功能。它通过**明确的文本指令 (Prompt Engineering)** 来引导模型，然后用一个**独立的解析器 (Parser)** 来处理模型的纯文本输出。
*   **工作流程：** 这是您在新例子中看到的方式。
    1.  **创建解析器**：`is_relevant_json_parser = JsonOutputParser(pydantic_object=Relevance)`
        *   这一步做了两件事：a) 创建了一个能将 JSON 字符串转换成 `Relevance` 对象的工具。b) 更重要的是，这个 `parser` 对象内部有一个方法可以生成“格式化指令”。

    2.  **生成并注入指令**：`partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()}`
        *   `get_format_instructions()` 会生成一段文本，这段文本会**详细地告诉** LLM 它应该如何构建 JSON，包括需要哪些键、值的类型是什么等等。
        *   这段生成的文本会被插入到你的 Prompt 模板中 `{format_instructions}` 的位置。
        *   **（揭秘！）`get_format_instructions()` 生成的内容大概是这样的：**
            ```text
            The output should be formatted as a JSON instance that conforms to the JSON schema below.

            As an example, for the schema {"properties": {"foo": {"title": "Foo", "type": "string"}}, "required": ["foo"]}
            the object {"foo": "bar"} is a well-formatted instance of the schema. The object {"properties": {"foo": "bar"}} is not.

            Here is the output schema:
            ```json
            {"properties": {"is_relevant": {"title": "Is Relevant", "description": "Whether the document is relevant to the query.", "type": "boolean"}, "explanation": {"title": "Explanation", "description": "An explanation of why the document is relevant or not.", "type": "string"}}, "required": ["is_relevant", "explanation"]}
            ```
            ```

    3.  **构建执行链**：`is_relevant_content_chain = is_relevant_content_prompt | fast_llm_client | is_relevant_json_parser`
        *   `prompt`：接收输入，并把上面那段指令填充进去，生成最终的完整 Prompt。
        *   `fast_llm_client`：接收这个长长的 Prompt，并生成一个**纯文本字符串**，它会“尽力”模仿你要求的 JSON 格式。
        *   `is_relevant_json_parser`：接收 LLM 返回的纯文本字符串，**尝试**将其解析成 JSON，然后验证并转换成 `Relevance` 对象。

*   **优点：**
    *   **通用性强**：**几乎适用于任何大语言模型**。只要这个模型能理解并遵循文本指令，你就可以用这个方法。
    *   **控制更精细**：你可以完全控制注入到 Prompt 里的指令，甚至可以手动修改 `get_format_instructions()` 返回的字符串。
*   **缺点：**
    *   **可靠性较低 (更脆弱)**：模型可能会“不听话”，比如忘记加逗号、在JSON前后添加一些多余的解释文字 (“Here is the JSON you requested: ...”)，这些都会导致 `JsonOutputParser` 解析失败并报错。
    *   **代码更冗长**：需要额外创建 `parser`、处理 `format_instructions`、并在链的末尾添加 `parser`。

---

### 总结与对比

| 特性 | 方法一 (`with_structured_output`) | 方法二 (`JsonOutputParser`) |
| :--- | :--- | :--- |
| **工作原理** | 依赖模型**原生**的工具调用能力 | 依赖**Prompt指令**和**后续解析** |
| **可靠性** | **非常高** | **中等**，依赖模型的“听话”程度 |
| **模型支持** | **有限** (仅支持的先进模型) | **广泛** (几乎所有模型) |
| **代码简洁性** | **非常简洁** | **相对冗长** |
| **Prompt** | 干净，不含格式化指令 | 包含大量关于JSON格式的指令 |

**为什么你的例子中会使用第二种方法？**

最可能的原因是 `fast_llm_client` 这个模型（比如 `llama3-70b-8192` 在某些平台上的实现）**不支持** `with_structured_output` 所需的工具调用功能。在这种情况下，开发者别无选择，只能退回到更通用、更经典的“Prompt + Parse”方法。

**最终建议：**

1.  **首选 `llm.with_structured_output()`**。只要你使用的模型支持，就应该优先使用这种方式，因为它更简单、更可靠。
2.  当你的模型**不支持**工具调用，或者你需要与一个非常基础、能力有限的模型交互时，**`JsonOutputParser` 就是你的最佳选择**。它为你提供了一条虽不完美但普遍适用的道路。

**经过查找**
llama3-70b-8192 在AIHUBMIX上确实不支持工具调用功能：
![alt text](image.png)


# 关键函数用法
提示词模版：
PromptTemplate(
    template=is_relevant_content_prompt_template,
    input_variables=["query", "context"],
    partial_variables={"format_instructions": is_relevant_json_parser.get_format_instructions()},
)
model.invoke(messages)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})