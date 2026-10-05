from langchain_core.messages import ToolMessage

def get_tool_messages(ai_msg, tools):
    if len(ai_msg.tool_calls) == 0:
        return []

    tool_messages = []
    for call in ai_msg.tool_calls:
        name = call["name"]
        args = call["args"]
        tool_id = call["id"]

        # 找到对应工具对象
        tool = next(t for t in tools if t.name == name)

        # 执行工具（不同工具可能用 invoke 或 run；LangChain tools一般是 invoke）
        observation = tool.invoke(args)

        tool_messages.append(
            ToolMessage(content=str(observation),
            tool_call_id=tool_id)
        )

    return tool_messages

def get_model_name(llm):
    for attr in ["model", "model_name", "model_id"]:
        if hasattr(llm, attr):
            val = getattr(llm, attr)
            if isinstance(val, str):
                return val
    return None