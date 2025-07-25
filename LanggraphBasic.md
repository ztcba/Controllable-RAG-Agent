## tips
定义的时候你不需要给它 state 参数，因为 LangGraph 框架会在未来替你调用它，并自动把当前的 state 传进去。

    你为什么不传 state？ 因为在 add_node 的时候，你只是在告诉 LangGraph “将来要调用这个函数”，而不是现在就调用它。

    state 从哪里来？ 它由 LangGraph 框架在运行时进行管理，并在调用每个节点函数时自动作为第一个参数传递进去。

    快速理解的关键是什么？ 牢牢记住“定义 vs 执行”的区别。add_node 是定义，stream/invoke 是执行。你的函数是在执行阶段被框架调用的，而不是在定义阶段被你调用的。




## agent
- agent执行示意
![alt text](agent.png)
Agent loop: the LLM selects tools and uses their outputs to fulfill a user request

- 代理使用一个语言模型，该模型期望以 messages 列表作为输入。因此，代理的输入和输出被存储为 messages 列表，并在代理state下的 messages 键中保存。

### 节点
在 LangGraph 中，节点是 Python 函数（同步或异步），它们接受以下参数：
- state : 图的状态
- config : 一个包含配置信息（如 thread_id ）和跟踪信息（如 tags ）的 RunnableConfig 对象
- runtime : 一个包含运行时 context 和其他信息（如 store 和 stream_writer ）的 Runtime 对象

与 NetworkX 类似，您使用 add_node 方法将这些节点添加到图中：

### 边
如果你总是想从节点 A 到节点 B，你可以直接使用 add_edge 方法。
`graph.add_edge("node_a", "node_b")`
如果你想要选择性地路由到 1 个或多个边缘（或选择性地终止），你可以使用 add_conditional_edges 方法。这个方法接受一个节点的名称和一个在执行该节点后调用的"路由函数"：
`graph.add_conditional_edges("node_a", routing_function)`
可以可选地提供一个字典，将 routing_function 的输出映射到下一个节点的名称。
`graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})

### 图

从本质上讲，LangGraph 将代理工作流程建模为图。你使用三个关键组件定义代理的行为：

    State: A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a TypedDict or Pydantic BaseModel.
    State ：一个共享的数据结构，表示你应用程序的当前快照。它可以是任何 Python 类型，但通常是 TypedDict 或 Pydantic BaseModel 。

    Nodes: Python functions that encode the logic of your agents. They receive the current State as input, perform some computation or side-effect, and return an updated State.
    Nodes : 编写你的智能体逻辑的 Python 函数。它们接收当前 State 作为输入，执行某些计算或副作用，并返回更新的 State 。

    Edges: Python functions that determine which Node to execute next based on the current State. They can be conditional branches or fixed transitions.
    Edges : 根据当前 State 确定执行下一个 Node 的 Python 函数。它们可以是条件分支或固定转换。

By composing Nodes and Edges, you can create complex, looping workflows that evolve the State over time. The real power, though, comes from how LangGraph manages that State. To emphasize: Nodes and Edges are nothing more than Python functions - they can contain an LLM or just good ol' Python code.
通过组合 Nodes 和 Edges ，你可以创建复杂、循环的工作流程，使 State 随时间演变。然而，真正的强大之处在于 LangGraph 如何管理 State 。要强调的是： Nodes 和 Edges 仅仅是 Python 函数——它们可以包含 LLM 或传统的 Python 代码。

In short: nodes do the work, edges tell what to do next.
简而言之：节点执行工作，边指明下一步该做什么。`

### prompt_template用法

https://blog.csdn.net/kljyrx/article/details/139296356
# 杂项
#### 调用链与高阶函数
当然！你提出的问题非常好，这正是理解像 LangGraph 这样的框架如何工作的核心。我们一步步来拆解，让你彻底摆脱“一头雾水”的感觉。

---

### 关键：快速理解的钥匙

在深入细节之前，先给你一把能快速理解这一切的“万能钥匙”。请记住这个核心思想：

**“区分定义和执行” (Distinguish between Definition and Execution)**

1.  **定义阶段 (Definition Phase)**：你看到的所有 `StateGraph(...)`, `.add_node(...)`, `.add_edge(...)` 代码，都**不是在运行**任何逻辑。它们只是在**构建一个蓝图**或**流程图**。就像用 Visio 画流程图一样，你正在告诉程序：“这里有一个节点叫 A，那里有一个节点叫 B，A 执行完后要去 B”。
2.  **执行阶段 (Execution Phase)**：只有当你调用 `.compile()` 创建应用，然后再调用 `.invoke()` 或 `.stream()` 时，这个蓝图才会被**激活**，数据（State）才开始按照你定义的路径在节点之间真正地流动起来。

当你能清晰地把代码分为“这是在画图纸”和“这是在按图纸施工”两部分时，一切都会变得清晰。

---

### 问题二：为什么能把函数名作为参数传进去？

我们先解决这个基础但至关重要的问题，因为它是一切的关键。

`qualitative_chunks_retrieval_workflow.add_node("keep_only_relevant_content", keep_only_relevant_content)`

这行代码能够工作，是因为 Python 的一个核心特性：**函数是“一等公民” (First-Class Citizens)**。

这意味着，函数在 Python 中与其他数据类型（如整数 `int`、字符串 `str`、列表 `list`）的地位是平等的。你可以：

1.  **将函数赋值给一个变量。**
2.  **将函数作为参数传递给另一个函数。** (这就是你的问题所在！)
3.  **将函数作为另一个函数的返回值。**

能接收其他函数作为参数的函数，我们称之为**高阶函数 (Higher-Order Function)**。这里的 `.add_node()` 就是一个高阶函数。

让我们看一个极简的例子：
```python
# 1. 定义两个简单的函数
def say_hello(name):
    print(f"你好, {name}!")

def say_goodbye(name):
    print(f"再见, {name}!")

# 2. 定义一个“高阶函数”，它接收一个函数作为参数
def greet(greeter_function, person_name):
    # 这里，greeter_function 是一个变量，它存储了传进来的函数
    print("准备执行问候...")
    # 执行传进来的函数
    greeter_function(person_name) 

# 3. 调用高阶函数，并把其他函数的名字（函数对象）传进去
greet(say_hello, "张三")    # 我们传递了 say_hello 函数本身
greet(say_goodbye, "李四") # 我们传递了 say_goodbye 函数本身

```
**输出:**
```
准备执行问候...
你好, 张三!
准备执行问候...
再见, 李四!
```




**回到你的代码：**

`add_node("节点名", 函数名)` 的工作原理和上面的 `greet` 函数一模一样。

*   它**不会立即执行** `keep_only_relevant_content` 函数。
*   它只是在内部的一个字典或类似结构中记录下来：`{"keep_only_relevant_content": <指向 keep_only_relevant_content 函数的内存地址>}`。
*   它把**函数本身**当作一张“待办事项卡片”存了起来。

等到未来**执行阶段**，当流程图走到名为 `"keep_only_relevant_content"` 的节点时，LangGraph 框架才会从它的记录中找到这张“卡片”，然后调用它：`keep_only_relevant_content(current_state)`。

---

### 问题一：解析这一系列的调用链条

现在我们用“区分定义和执行”的钥匙，来解锁你复杂的调用链。

这是一个“**大图套小图**”或者叫“**嵌套工作流**”的结构。可以把它想象成一个公司的组织架构：

*   **`create_agent()`**：创建的是**总指挥部 (CEO)**，它的状态是 `PlanExecute`，包含了整个任务的所有信息。
*   **`run_qualitative_chunks_retrieval_workflow`** 等函数：它们本身是总指挥部图谱中的**一个节点**，但这个节点的任务是去调用一个**专门的部门 (Sub-workflow)** 来完成一项具体工作（比如“信息检索部”）。

我们来梳理一下这个链条：

#### **第一层：定义“信息检索部”工作流 (子图)**

1.  **`create_qualitative_retrieval_book_chunks_workflow_app()`**
    *   **阶段**：**定义阶段**。
    *   **任务**：定义一个专门负责“从书中检索切片并筛选”的子流程。
    *   **状态蓝图**：`QualitativeRetrievalGraphState` (一个比 `PlanExecute` 更简单的状态)。
    *   **节点**：
        *   `retrieve_chunks_context_per_question`: 检索。
        *   `keep_only_relevant_content`: 筛选相关内容。
    *   **流程**：`检索 -> 筛选 -> (条件判断) -> 结束或重新筛选`。
    *   **产出**：返回一个编译好的、可执行的子应用 `qualitative_chunks_retrieval_workflow_app`。

#### **第二层：定义“总指挥部”工作流 (主图)**

2.  **`create_agent()`**
    *   **阶段**：**定义阶段**。
    *   **任务**：定义整个 Agent 的总工作流程。
    *   **状态蓝图**：`PlanExecute` (包含所有信息的全局状态)。
    *   **节点**：它定义了很多节点，我们重点看一个：
        *   `agent_workflow.add_node("retrieve_chunks", run_qualitative_chunks_retrieval_workflow)`
        *   这里，它将 `"retrieve_chunks"` 这个节点名与 `run_qualitative_chunks_retrieval_workflow` 这个**函数**关联起来。

#### **第三层：作为节点的“包装函数”**

3.  **`run_qualitative_chunks_retrieval_workflow(state)`**
    *   **阶段**：这个函数本身是在**执行阶段**被调用的。当主图（Agent）的流程走到 `"retrieve_chunks"` 节点时，LangGraph 框架会调用它。
    *   **任务**：它是一个“**适配器**”或“**包装器**”。它的职责是连接主图和子图。
    *   **步骤**：
        1.  从主图的全局状态 `state` (类型是 `PlanExecute`) 中，提取出子图需要的输入信息，例如 `question = state["query_to_retrieve_or_answer"]`。
        2.  创建一个 `inputs` 字典，这是子图的入口参数。
        3.  **调用并执行子图**：`qualitative_chunks_retrieval_workflow_app.stream(inputs)`。
        4.  等待子图执行完毕，并拿到子图的最终输出 `output`。
        5.  将子图的输出结果（如 `output['relevant_context']`）**更新回主图的全局状态**中：`state["aggregated_context"] += ...`。
        6.  返回更新后的 `state`，以便主图继续接下来的流程（比如 `replan`）。

### 总结与调用流程图

**定义时 (画蓝图):**
`create_agent()` -> `add_node("retrieve_chunks", run_qualitative_chunks_retrieval_workflow)`
    (内部) `create_qualitative_..._app()` -> `add_node("keep_only_relevant_content", keep_only_relevant_content)`

**执行时 (跑流程):**

1.  `plan_and_execute_app.invoke(...)` 启动 **主图 (Agent)**。
2.  主图流程走到 `"task_handler"` 节点。
3.  `"task_handler"` 决定下一步是 `"retrieve_chunks"`。
4.  LangGraph 框架调用与 `"retrieve_chunks"` 节点绑定的函数：`run_qualitative_chunks_retrieval_workflow(currentState)`。
5.  在 `run_...` 函数内部：
    a.  提取 `question`。
    b.  调用 **子图** `qualitative_chunks_retrieval_workflow_app.stream(...)`。
6.  子图开始执行自己的流程：
    a.  走到 `"retrieve_chunks_context_per_question"` 节点，调用 `retrieve_chunks_context_per_question()` 函数。
    b.  走到 `"keep_only_relevant_content"` 节点，调用 `keep_only_relevant_content()` 函数。
    c.  ...直到子图结束，返回结果 `output`。
7.  `run_...` 函数拿到 `output`，更新 `currentState`。
8.  `run_...` 函数返回更新后的 `currentState` 给主图框架。
9.  主图根据流程，继续走向下一个节点，如 `"replan"`。

通过这种“大图套小图”的模式，你可以将一个复杂的任务分解成多个可独立维护和测试的、更简单的子任务，极大地提高了代码的模块化和复用性。

## langgraph核心
好的，我们正式进入高级阶段。当你发现一个任务无法用一条直线（即使是包含并行处理的直线）来描述时，你就需要 LangGraph 了。

LangChain 的 LCEL 擅长构建**有向无环图 (DAGs)**，数据从一端流入，另一端流出。但如果你需要**循环 (loops)**、**条件判断 (branching)**，或者更复杂的、可以根据自身状态改变行为的**智能体 (Agent)**，那么 LangGraph 就是你需要的工具。

---

### LangGraph 高级：构建有状态的多步智能体

#### **核心思想：将应用建模为“状态机”**

忘掉线性的“链”，把你的应用想象成一个**流程图**或**状态机**。

*   **状态 (State)**: 在任何时刻，你的应用都有一个明确的“状态”，它包含了所有需要的信息，比如用户的原始问题、检索到的文档、已经生成的草稿、历史对话等。这是一个贯穿始终的共享“记忆”。
*   **节点 (Nodes)**: 流程图中的每一个“框”，代表一个操作单元。这个操作会接收当前的**状态**，执行一些工作（比如调用模型、调用工具），然后返回一个**更新**，这个更新会被合并回主状态中。
*   **边 (Edges)**: 流程图中的“箭头”，决定了在一个节点完成后，接下来应该去哪个节点。这可以是固定的（A 之后永远是 B），也可以是**有条件的**（如果 A 的结果是 X，就去 B；如果是 Y，就去 C）。

LangGraph 的本质就是让你用 Python 来定义这个**状态**、这些**节点**和连接它们的**边**，然后把它编译成一个可执行的图。

#### **核心类与组件详解**

##### **1. 状态 (State)**

*   **概念**: 定义你的图在整个运行过程中需要跟踪的所有数据。
*   **实现方式**: 通常使用 Python 的 `TypedDict` 来定义一个结构化的字典。这提供了类型提示，让代码更清晰。你也可以用 Pydantic `BaseModel`。
*   **关键点**:
    *   状态是**累加**的。每个节点返回的字典会更新（而不是替换）现有的状态。你可以通过配置来改变这个行为，但默认是累加。
        * 在 LangGraph 中，状态的累加行为遵循 "字典合并" 的逻辑，但具体到不同类型的字段（如列表、基本类型）会有不同表现，这取决于节点返回值的结构：对于列表类型（如 messages）：
        当节点返回一个新的列表时，LangGraph 默认会将其追加到原有列表后，而不是替换。
        这是因为在对话场景中，消息列表通常需要保留完整的对话历史（如聊天记录），追加是更符合直觉的行为。
        例如：原有["你好"] + 节点返回["我很好"] → 合并为["你好", "我很好"]
        对于基本类型（如 count）：
        当节点返回一个新的基本类型值（数字、字符串等）时，会直接覆盖原有值。
        这是因为基本类型通常表示 "当前状态值"（如计数器、开关状态），新值自然应该替代旧值。
        例如：原有count=1 + 节点返回count=2 → 合并为count=2。
        这种行为设计是为了适配对话类应用的常见需求 —— 既需要保留历史信息（如消息列表），又需要更新当前状态（如计数、流程标记）。
        如果需要修改默认行为（例如让列表也被替换），可以通过配置StateGraph的merge_strategy参数自定义合并逻辑。

    *   所有节点共享同一个状态对象。
*   **示例**:
    ```python
    from typing import TypedDict, List
    from langchain_core.documents import Document

    class AgentState(TypedDict):
        question: str           # 用户的原始问题
        documents: List[Document] # 检索到的文档
        generation: str         # LLM 生成的当前答案
        iterations: int         # 记录循环次数
    ```

##### **2. 图 (Graph)**

*   **核心类**: `langgraph.graph.StateGraph`
*   **`__init__(self, state_schema)`**:
    *   **接收参数**: `state_schema` (一个 `TypedDict` 或 `BaseModel` 类)。这是你创建图的第一步，告诉 LangGraph 你的状态长什么样。
    *   **返回对象**: 一个 `StateGraph` 实例，这是你的“画板”，你将在上面添加节点和边。
*   **示例**:
    ```python
    from langgraph.graph import StateGraph

    # 使用我们上面定义的状态模式来初始化图
    workflow = StateGraph(AgentState)
    ```

##### **3. 节点 (Nodes)**

*   **概念**: 一个接收当前状态并执行操作的 Python 函数或任何 `Runnable`。
    * Runnable:在 LangChain/LangGraph 的语境中，"Runnable" 可以翻译为 「可运行对象」 或 「可执行单元」，从理解层面来说，将其理解为 「具有统一运行接口的组件」 更为贴切。

    这个概念的核心不在于字面翻译，而在于它的本质：

        它是一个「能干活的组件」—— 可以接收输入、执行特定操作（比如调用模型、运行函数、调用工具等）、然后输出结果。
        它有「标准化的接口」—— 无论是什么类型的组件（模型、函数、工具链等），只要是 Runnable，就可以用相同的方式（比如 invoke() 方法）去调用，不用关心内部实现。


    打个比方，Runnable 就像生活中的「电器接口」：不管是洗衣机、电视还是手机充电器，只要符合国标接口，都能插在同一个插座上使用。在这里，「插座」就是统一的调用方式，「电器」就是各种不同功能的 Runnable 组件。

    所以，与其纠结翻译，不如记住：只要一个东西能被 LangGraph 调用（接收输入、产生输出），并且遵循了统一的调用规范，它就是 Runnable。这种设计让不同组件可以无缝协作，是 LangChain 生态灵活性的重要基础。

    常见的 Runnable 类型：

    - Python 函数
    普通的 Python 函数只要接收状态作为输入并返回新的状态数据，就可以作为节点（本质上是被包装成了 Runnable）。
    - LLM 模型（如 ChatOpenAI）
    - LangChain 中封装的大语言模型对象（如 ChatOpenAI、ChatAnthropic 等）都是 Runnable。例如，你可以直接将一个模型实例作为节点，它会自动处理输入的状态（如提取消息列表）并返回模型的响应。
    

 

    from langchain.chat_models import ChatOpenAI

    (这是一个 Runnable，可以直接作为节点)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")




    - 工具调用链（Toolchain）
    由工具（Tools）和模型组成的调用链（如 create_tool_calling_chain 创建的链）也是 Runnable。这类节点可以处理复杂的逻辑，比如根据状态决定是否调用工具，并将工具结果返回给状态。
    - 自定义 Runnable
    你可以通过继承 Runnable 类或使用 @runnable 装饰器定义自己的 Runnable，实现更复杂的逻辑。
*   **函数签名**:
    *   **接收参数**: `state` (一个字典，其结构与你定义的 `AgentState` 一致)。
    *   **返回对象**: 一个字典，包含了你想要**更新或添加**到状态中的键值对。**注意：你不需要返回完整的状态，只需要返回你想改变的部分。**
*   **添加节点的方法**: `graph.add_node(name: str, action: callable)`
    *   `name`: 节点的唯一字符串名称，用于后续连接边。
    *   `action`: 你的节点函数。
*   **示例**:
    ```python
    def retrieve_documents(state: AgentState) -> dict:
        print("---NODE: RETRIEVE DOCUMENTS---")
        question = state["question"]
        # retriever 是一个已经配置好的 LangChain Retriever
        documents = retriever.invoke(question)
        return {"documents": documents, "iterations": state.get("iterations", 0) + 1}

    # 将这个函数添加为图中的一个节点
    workflow.add_node("retriever", retrieve_documents)
    ```

##### **4. 边 (Edges)**

这是 LangGraph 最强大的部分。

*   **A) 普通边 (Normal Edges)**
    *   **概念**: 无条件地将一个节点连接到另一个节点。
    *   **添加方法**: `graph.add_edge(start_node_name: str, end_node_name: str)`
    *   `start_node_name`: 边的起始节点名称。
    *   `end_node_name`: 边的结束节点名称。

*   **B) 条件边 (Conditional Edges)**
    *   **概念**: 根据一个“路由函数”的输出来决定下一步去哪个节点。这是实现**判断和循环**的关键。
    *   **添加方法**: `graph.add_conditional_edges(source_node_name: str, router_function: callable, path_map: dict)`
        *   `source_node_name`: 做出决策的节点。路由函数会在该节点执行**之后**运行。
        *   `router_function`: 一个决策函数。它接收当前的状态 `state`，并返回一个**字符串**。这个字符串决定了走哪条路。
        *   `path_map`: 一个字典，`key` 是路由函数可能返回的字符串，`value` 是对应的下一个节点的名称。
    *   **示例 (续上)**:
        ```python
        def should_continue(state: AgentState) -> str:
            print("---DECISION: SHOULD CONTINUE?---")
            if state["iterations"] > 3:
                print("-> Limit reached, finishing.")
                return "end"
            else:
                print("-> Continue generation.")
                return "continue"

        # 在 "retriever" 节点之后，我们调用 should_continue 来做决定
        workflow.add_conditional_edges(
            "retriever",  # 决策的源头
            should_continue, # 路由函数
            {
                "continue": "generator", # 如果返回 "continue"，就去 "generator" 节点
                "end": END               # 如果返回 "end"，就结束流程
            }
        )
        ```

*   **C) 入口和出口**
    *   `graph.set_entry_point(node_name: str)`: 指定图从哪个节点开始执行。
    *   `END`: 这是一个特殊的常量，用在边的定义中，表示流程结束。`from langgraph.graph import END`。

##### **5. 编译和运行**

*   **概念**: 当你定义完所有的节点和边后，你需要将这个“定义”编译成一个可执行的 `Runnable` 对象。
*   **方法**: `graph.compile()`
    *   **返回对象**: 一个 `CompiledGraph` 对象，它和 LCEL 链一样，拥有 `.invoke()`, `.stream()`, `.batch()` 等方法。
*   **示例**:
    ```python
    # 假设我们已经添加了 retriever, generator 等所有节点和边

    # 设置入口点
    workflow.set_entry_point("retriever")

    # 编译图
    app = workflow.compile()

    # 运行图
    initial_state = {"question": "LangGraph 中的循环怎么实现？", "iterations": 0}
    for event in app.stream(initial_state):
        for key, value in event.items():
            print(f"--- Event from node: {key} ---")
            print(value)
            print("\n")
    ```

#### **总结一下 LangGraph 的核心用法**

1.  **定义状态 `State`**: 用 `TypedDict` 规划好你的应用需要的所有数据。
2.  **实例化图 `StateGraph`**: `graph = StateGraph(State)`。
3.  **定义节点 `Nodes`**: 编写一系列函数，每个函数都接收 `state` 并返回一个要更新的 `dict`。
4.  **添加节点**: 使用 `graph.add_node()` 将你的函数注册到图中。
5.  **定义并添加边 `Edges`**:
    *   用 `graph.add_edge()` 连接确定的步骤。
    *   用 `graph.add_conditional_edges()` 来实现判断和循环，这是 LangGraph 的精髓。
6.  **设置起止点**: 使用 `graph.set_entry_point()` 和 `END` 来确定流程的开始和结束。
7.  **编译 `compile()`**: 调用 `graph.compile()` 将定义转化为一个可执行的 `app`。
8.  **运行 `invoke()`/`stream()`**: 像调用普通 LCEL 链一样调用 `app`，传入初始状态。

LangGraph 的学习曲线比 LCEL 要陡峭，因为它引入了更复杂的编程范式。但一旦你掌握了**状态-节点-边**这个核心模型，你就能构建出远比简单 RAG 强大的、具备复杂逻辑和自主决策能力的应用程序。