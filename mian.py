import os
import dashscope
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dashscope import Generation

# ===================== 1. 加载环境变量 =====================
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise ValueError("请配置 DASHSCOPE_API_KEY")

def call_llm(
    question: str,
    temperature: float = 0.5,
    top_p: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """
    封装通义千问调用函数
    :param question: 用户问题
    :param temperature: 随机性
    :param top_p: 采样范围
    :param max_tokens: 最大输出长度
    :return: 模型回答
    """
    try:
        response = Generation.call(
            model="qwen-turbo",  # 免费轻量版模型
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            result_format="message"
        )
        # 解析回答
        return response.output.choices[0].message.content
    
    except Exception as e:
        return f"调用失败：{str(e)}"

# ===================== 任务1：基础调用 =====================
if __name__ == "__main__":
    print("=== 基础调用 ===")
    question = "什么是RAG检索增强生成？"
    answer = call_llm(question)
    print(f"问题：{question}")
    print(f"回答：{answer}\n")

# ===================== 任务2：参数测试实验（核心必做） =====================
    print("=== 参数对比测试：Temperature 0.1（精准） VS 0.9（创意） ===")
    test_question = "写一句关于春天的句子"

    # 参数1：超低温度 → 严谨、固定、标准
    print("\n【Temperature=0.1 精准模式】")
    print(call_llm(test_question, temperature=0.1))

    # 参数2：高温度 → 创意、发散、生动
    print("\n【Temperature=0.9 创意模式】")
    print(call_llm(test_question, temperature=0.9))