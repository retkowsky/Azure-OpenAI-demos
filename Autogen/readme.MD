# AutoGen

The AutoGen ecosystem provides everything you need to create AI agents, especially multi-agent workflows -- framework, developer tools, and applications.

The framework uses a layered and extensible design. Layers have clearly divided responsibilities and build on top of layers below. This design enables you to use the framework at different levels of abstraction, from high-level APIs to low-level components.

Core API implements message passing, event-driven agents, and local and distributed runtime for flexibility and power. It also support cross-language support for .NET and Python. AgentChat API implements a simpler but opinionated API for rapid prototyping. This API is built on top of the Core API and is closest to what users of v0.2 are familiar with and supports common multi-agent patterns such as two-agent chat or group chats. Extensions API enables first- and third-party extensions continuously expanding framework capabilities. It support specific implementation of LLM clients (e.g., OpenAI, AzureOpenAI), and capabilities such as code execution.

<img src="https://camo.githubusercontent.com/3c3fd32e30a086037ba14570534c4943ac7d6b24953ce821a4321fd616975a4a/68747470733a2f2f6769746875622e636f6d2f6d6963726f736f66742f6175746f67656e2f7261772f6d61696e2f6175746f67656e2d6c616e64696e672e6a7067">

- https://microsoft.github.io/autogen/stable/
- https://github.com/microsoft/autogen
