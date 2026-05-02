import os
import dashscope
from dashscope import Generation
from dotenv import load_dotenv

# 加载配置
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ===================== 1. 通用防幻觉Prompt（必套所有对话） =====================
ANTI_HALLUCINATION = """
【核心约束】
1. 不知道就说「我不知道」，禁止编造、禁止瞎猜
2. 只基于常识或给定内容回答，不添加额外信息
3. 回答简洁、不啰嗦、紧扣问题
"""

# ===================== 2. 带记忆的多轮对话类（核心封装） =====================
class MemoryLLM:
    def __init__(self, max_history: int = 5, model: str = "qwen-turbo"):
        """
        初始化带记忆的对话类
        :param max_history: 最大保留历史轮数
        :param model: 调用的模型
        """
        self.history = []
        self.max_history = max_history
        self.model = model

    def _trim_history(self):
        """自动裁剪历史，避免超长"""
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def chat(self, user_input: str, system_prompt: str = ANTI_HALLUCINATION) -> str:
        """
        带记忆的对话函数
        :param user_input: 用户当前输入
        :param system_prompt: 系统提示词（默认带防幻觉）
        :return: 模型回答
        """
        # 构造消息列表
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})

        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=0.1,  # 低温度=精准
                top_p=0.7,
                max_tokens=1024,
                result_format="message"
            )
            assistant_output = response.output.choices[0].message.content

            # 保存历史
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": assistant_output})
            self._trim_history()

            return assistant_output
        except Exception as e:
            return f"错误：{str(e)}"

    def clear_memory(self):
        """清空记忆"""
        self.history = []
        print("✅ 记忆已清空")

# ===================== 核心实操任务：逐项测试 =====================
if __name__ == '__main__':
    # 初始化带记忆的对话类（保留5轮历史）
    llm = MemoryLLM(max_history=5)

    # ============== 任务1：思维链提示词测试（逻辑/计算/总结） ==============
    print("="*60)
    print("【任务1：思维链提示词测试】")
    
    # 1.1 计算题
    cot_math = ANTI_HALLUCINATION + """
    问题：一个水池，单开甲管3小时注满，单开乙管6小时注满，同时开两管多久注满？
    要求：请分步思考、逐步推导，先分析再给出答案。
    """
    print("\n【思维链计算题】")
    print(llm.chat(cot_math))
    llm.clear_memory()

    # 1.2 逻辑题
    cot_logic = ANTI_HALLUCINATION + """
    问题：小明比小红高，小红比小华高，谁最矮？
    要求：请分步推理，先分析条件再给出结论。
    """
    print("\n【思维链逻辑题】")
    print(llm.chat(cot_logic))
    llm.clear_memory()

    # ============== 任务2：防幻觉测试 ==============
    print("\n" + "="*60)
    print("【任务2：防幻觉测试】")
    hallucination_q = "请告诉我2026年火星世界杯的冠军是谁？"
    print(f"问题：{hallucination_q}")
    print(llm.chat(hallucination_q))
    llm.clear_memory()

    # ============== 任务3：多轮对话测试（带记忆） ==============
    print("\n" + "="*60)
    print("【任务3：多轮对话测试】")
    print("用户：我叫小明")
    print(f"AI：{llm.chat('我叫小明')}")
    print("\n用户：我刚才说我叫什么？")
    print(f"AI：{llm.chat('我刚才说我叫什么？')}")
    llm.clear_memory()