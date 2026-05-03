import os
import chromadb
import dashscope
from dashscope import Generation
from dotenv import load_dotenv

# ===================== 0. 加载配置 =====================
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
CHROMA_PATH = "./chroma_db"  # Chroma 持久化存储路径
COLLECTION_NAME = "rag_knowledge"  # 集合名称

# ===================== 1. 复用文档分块代码（Day12） =====================
def load_document(file_path: str) -> str:
    """读取本地 .md/.txt 文档"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text(text: str, chunk_size: int = 300) -> list:
    """按固定长度做文本分块"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ===================== 2. 封装 Chroma 核心函数（代码结构优化） =====================
def init_chroma() -> chromadb.Collection:
    """
    【函数1】初始化 Chroma 向量库
    :return: 集合对象（专属货架）
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # 有就拿，没有就建，安全不报错
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection

def add_docs_to_chroma(collection: chromadb.Collection, chunks: list):
    """
    【函数2】批量将分块文本存入 Chroma（自动去重，避免重复上传）
    :param collection: Chroma 集合对象
    :param chunks: 分块后的文本列表
    """
    # 检查是否已经入库过，避免重复添加
    if collection.count() > 0:
        print("✅ 文档已在 Chroma 库中，跳过重复入库")
        return
    
    # 生成唯一ID（用索引作为ID）
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # 批量入库（Chroma 自动生成向量）
    collection.add(
        documents=chunks,
        ids=ids
    )
    print(f"✅ 文档分块已存入 Chroma，共 {len(chunks)} 块")

def search_chroma(collection: chromadb.Collection, query: str, top_k: int = 3) -> list:
    """
    【函数3】语义相似度检索，返回最相关的 TOP-K 文档片段
    :param collection: Chroma 集合对象
    :param query: 用户问题
    :param top_k: 返回条数
    :return: 相关文档片段列表
    """
    results = collection.query(
        query_texts=query,
        n_results=top_k
    )
    return results['documents'][0]

# ===================== 3. RAG 问答整合（对接 Chroma 检索结果） =====================
def rag_qa_chroma(collection: chromadb.Collection, query: str) -> str:
    """
    基于 Chroma 的 RAG 完整问答
    :param collection: Chroma 集合对象
    :param query: 用户问题
    :return: 大模型回答
    """
    # 3.1 检索最相关的文档片段
    relevant_chunks = search_chroma(collection, query, top_k=3)
    context = "\n".join(relevant_chunks)
    
    # 3.2 拼接提示词（检索增强 + 防幻觉约束）
    system_prompt = f"""
    你是专业的知识库问答助手。
    【核心约束】
    1. 只根据下面的【检索到的文档内容】回答问题，禁止编造、禁止瞎猜
    2. 如果文档里没有相关内容，直接说「我不知道，文档里没有相关内容」
    3. 回答简洁、紧扣问题、不啰嗦
    
    【检索到的文档内容】
    {context}
    """
    
    # 3.3 调用大模型生成回答
    response = Generation.call(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        max_tokens=1024,
        result_format="message"
    )
    return response.output.choices[0].message.content

# ===================== 4. 测试验证（必做） =====================
if __name__ == '__main__':
    # 4.1 初始化 Chroma
    print("="*60)
    print("【1/4】初始化 Chroma 向量库...")
    collection = init_chroma()
    
    # 4.2 加载并分块文档（仅第一次运行会入库，后续跳过）
    print("\n【2/4】加载并分块本地文档...")
    doc_text = load_document("knowledge.md")
    chunks = split_text(doc_text, chunk_size=300)
    add_docs_to_chroma(collection, chunks)
    
    # 4.3 测试专业问题（验证检索精准度）
    print("\n【3/4】RAG 问答测试...")
    print("="*60)
    
    # 测试1：文档内专业问题
    test_q1 = "RAG的核心作用是什么？"
    print(f"\n问题1（文档内）：{test_q1}")
    print(f"回答：{rag_qa_chroma(collection, test_q1)}")
    
    # 测试2：文档外问题（防幻觉）
    test_q2 = "2026年火星世界杯冠军是谁？"
    print(f"\n问题2（文档外）：{test_q2}")
    print(f"回答：{rag_qa_chroma(collection, test_q2)}")
    
    # 4.4 持久化验证提示
    print("\n" + "="*60)
    print("💡 持久化验证：现在关闭程序，重新运行，直接提问仍能检索到内容！")