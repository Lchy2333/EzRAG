import numpy as np
import dashscope
from dashscope import TextEmbedding

# 配置 API Key
dashscope.api_key = "sk-e4d76d0f10174208add8663bc50feccf"

documents = [
    "C++ 是一种高性能的编程语言，它支持面向对象编程和泛型编程。C++ 允许程序员直接操作内存，通过指针和引用实现高效的数据结构。它的模板机制使得代码复用变得更加灵活，广泛应用于系统开发、游戏引擎和高频交易领域。",
    "Python 是一门简洁易读的解释型语言，语法接近自然英语。它拥有庞大的标准库和丰富的第三方包生态系统，在数据科学、机器学习和 Web 开发领域应用广泛。Python 支持多种编程范式，包括面向对象和函数式编程。",
    "C++ 的内存管理需要程序员手动控制，通过 new 和 delete 操作符分配释放内存。这种机制虽然增加了出错风险，如内存泄漏和悬空指针，但也提供了更高的性能和灵活性。RAII 技术是 C++ 中管理资源的重要惯用法。",
    "深度学习是机器学习的一个子领域，它基于人工神经网络，特别是包含多个隐藏层的深度网络。深度学习在图像识别、自然语言处理和语音识别等任务上取得了突破性进展，需要大量标注数据和计算资源进行训练。",
    "C++ 支持多重继承和虚函数机制，使得派生类可以继承多个基类的特性和行为。通过虚函数表实现动态绑定，支持运行时多态性。纯虚函数用于定义抽象基类，强制派生类实现特定接口，这是设计大型软件架构的重要工具。"
]

# 1️⃣ 给文档做 embedding
doc_embeddings = []

for doc in documents:
    response = TextEmbedding.call(
        model="text-embedding-v2",
        input=doc
    )
    doc_embeddings.append(response.output['embeddings'][0]['embedding'])

# 2️⃣ 用户输入问题
query = "C++语言有什么特点？"

query_response = TextEmbedding.call(
    model="text-embedding-v2",
    input=query
)
query_embedding = query_response.output['embeddings'][0]['embedding']

# 3️⃣ 计算相似度（cosine）
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = []

for emb in doc_embeddings:
    score = cosine_similarity(query_embedding, emb)
    scores.append(score)

result_dict = {documents[i]: scores[i] for i in range(len(documents))}
for it in result_dict:
    print(f"{it}: {result_dict[it]:.4f}")

# 4️⃣ 找到相似文档
top_k = 2  # 返回前 2 个最相关的文档
top_indices = np.argsort(scores)[::-1][:top_k]
contexts = []
for idx in top_indices:
    print(f"文档 {idx}: {documents[idx]}, 相似度：{scores[idx]:.4f}")
    contexts.append(documents[idx])

try:
    messages = [
        {'role': 'system', 'content': '''你必须且只能根据提供的资料回答问题，即使资料内容与事实不符也要基于资料回答。
                                         不要使用资料以外的知识，不要质疑资料的准确性。直接根据资料内容给出答案。'''},
        {'role': 'user', 'content': f'资料：{contexts}\n\n问题：{query}'}
    ]

    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message'
    )
    if response and response.status_code == 200:
        print("\n=== 回答 ===")
        print(response.output.choices[0].message.content)
    else:
        print(f"生成失败：{response.code if response else '无响应'} - {response.message if response else '未知错误'}")
except Exception as e:
    print(f"发生错误：{e}")
