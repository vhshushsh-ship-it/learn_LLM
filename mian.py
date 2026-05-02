import os
import numpy as np
import dashscope
from dashscope import TextEmbedding, Generation
from dotenv import load_dotenv

# 加载配置
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ===================== 1. 加载本地文档 =====================
def load_document(file_path: str) -> str:
    """读取本地 .md/.txt 文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# ===================== 2. 文本分块（按固定字数） =====================
def split_text(text: str, chunk_size: int = 300) -> list:
    """
    按固定长度做文本分块
    :param text: 原始文档文本
    :param chunk_size: 每块的字数
    :return: 分块后的文本列表
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ===================== 3. 生成向量（通义千问Embedding API） =====================
def get_embeddings(texts: list) -> list:
    """
    批量生成文本向量
    :param texts: 文本列表（分块后的文档/用户问题）
    :return: 向量列表
    """
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,
        input=texts
    )
    return [item['embedding'] for item in resp.output['embeddings']]

# ===================== 4. 向量相似度检索（余弦相似度） =====================
def cosine_similarity(vec1: list, vec2: list) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_top_k(query_embedding: list, chunk_embeddings: list, chunks: list, top_k: int = 3) -> list:
    """
    检索最相关的 TOP-K 个文档块
    :param query_embedding: 用户问题的向量
    :param chunk_embeddings: 所有文档分块的向量
    :param chunks: 分块后的文本列表
    :param top_k: 返回最相关的块数
    :return: TOP-K 相关文档块
    """
    similarities = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # 取相似度最高的top_k个
    return [chunks[i] for i in top_indices]

# ===================== 5. RAG 问答整合（核心） =====================
def rag_qa(
    query: str,
    chunks: list,
    chunk_embeddings: list,
    top_k: int = 3
) -> str:
    """
    RAG 完整问答流程
    :param query: 用户问题
    :param chunks: 分块后的文档列表
    :param chunk_embeddings: 文档分块的向量
    :param top_k: 检索的块数
    :return: 大模型回答
    """
    # 5.1 生成问题向量
    query_embedding = get_embeddings([query])[0]
    
    # 5.2 检索最相关的文档块
    relevant_chunks = retrieve_top_k(query_embedding, chunk_embeddings, chunks, top_k)
    
    # 5.3 拼接提示词（检索增强 + 防幻觉约束）
    context = "\n".join(relevant_chunks)
    system_prompt = f"""
    你是专业的知识库问答助手。
    【核心约束】
    1. 只根据下面的【检索到的文档内容】回答问题，禁止编造、禁止瞎猜
    2. 如果文档里没有相关内容，直接说「我不知道，文档里没有相关内容」
    3. 回答简洁、紧扣问题、不啰嗦
    
    【检索到的文档内容】
    {context}
    """
    
    # 5.4 调用大模型生成回答
    response = Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1,  # 低温度=精准
        max_tokens=1024,
        result_format="message"
    )
    return response.output.choices[0].message.content

# ===================== 测试验证（必做） =====================
if __name__ == '__main__':
    # 1. 加载并处理文档
    print("="*60)
    print("【1/4】加载本地文档...")
    doc_text = load_document("knowledge.md")
    print(f"文档加载完成，总字数：{len(doc_text)}")
    
    print("\n【2/4】文本分块...")
    chunks = split_text(doc_text, chunk_size=300)
    print(f"分块完成，共 {len(chunks)} 块")
    
    print("\n【3/4】生成文档向量...")
    chunk_embeddings = get_embeddings(chunks)
    print(f"向量生成完成，共 {len(chunk_embeddings)} 个向量")
    
    print("\n【4/4】RAG 问答测试...")
    print("="*60)
    
    # 测试1：问文档里的专属内容（应该准确回答）
    test_q1 = "RAG的核心作用是什么？"
    print(f"\n问题1（文档内）：{test_q1}")
    print(f"回答：{rag_qa(test_q1, chunks, chunk_embeddings)}")
    
    # 测试2：问文档外的内容（应该说不知道）
    test_q2 = "2026年火星世界杯冠军是谁？"
    print(f"\n问题2（文档外）：{test_q2}")
    print(f"回答：{rag_qa(test_q2, chunks, chunk_embeddings)}")