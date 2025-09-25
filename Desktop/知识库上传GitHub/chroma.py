# 导入Chroma向量数据库的主库
import chromadb
# 从Chroma库中导入配置设置
from chromadb.config import Settings
# 从SentenceTransformers库中导入SentenceTransformer(用于将文本转换为向量)
from sentence_transformers import SentenceTransformer
# 从LangChain库中导入RecursiveCharacterTextSplitter(用于将文本分割为小块)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 从LangChain库中导入TextLoader, PyPDFLoader, Docx2txtLoader(用于加载文本文件)
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
# 尝试引入 Excel 加载器（保持简单，仅支持该加载器）
try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
except Exception:
    UnstructuredExcelLoader = None
# 导入pandas用于Excel文件处理
import pandas as pd
# 导入os库(用于操作文件和目录)
import os
# 导入uuid库(用于生成唯一标识符)
import uuid
# 导入datetime库(用于生成时间戳)
from datetime import datetime

# dotenv库用于管理环境变量
# 从dotenv库中导入load_dotenv(用于加载环境变量)
from dotenv import load_dotenv

## 数据库初始化

### 1. 创建 Chroma 客户端

def init_chroma_db():
    """初始化 Chroma 数据库"""
    # 创建持久化本地向量数据库
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db",  # 数据库存储路径
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 获取或创建集合
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"description": "知识库文档集合"}
    )
    
    return collection  # 返回已连接的集合对象，供后续添加与检索
    #collection = chroma.init_chroma_db()返回给text.py的collection对象

### 2. 初始化嵌入模型

def init_embedding_model():
    """初始化嵌入模型"""
    # 尝试加载环境变量文件，如果不存在则忽略
    try:
        load_dotenv("./.env")
    except:
        pass  # 如果.env文件不存在，继续使用默认值
    
    # 获取模型名称，如果没有设置则使用默认模型
    model_name = os.getenv("modelname", "all-MiniLM-L6-v2")
    
    try:
        # 初始化嵌入模型，明确指定使用CPU设备
        model = SentenceTransformer(model_name, device='cpu')
        return model  # 返回已初始化的嵌入模型，供后续生成向量使用
    except Exception as e:
        import streamlit as st
        st.write(f"模型初始化失败: {e}")
        # 如果指定模型失败，尝试使用默认模型
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            st.write("使用默认模型 all-MiniLM-L6-v2")
            return model
        except Exception as e2:
            st.write(f"默认模型也初始化失败: {e2}")
            raise e2

## 文件处理功能

### 1. 文件加载器

def load_document(file_path, file_type):
    """根据文件类型加载文档"""
    try:
        if file_type in ['txt', 'md', 'text/plain']:
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type in ['pdf','PDF','application/pdf']:
            loader = PyPDFLoader(file_path)
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
        elif file_type in ['xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            # 首先尝试使用UnstructuredExcelLoader
            if UnstructuredExcelLoader is not None:
                try:
                    loader = UnstructuredExcelLoader(file_path)
                    documents = loader.load()
                    return documents
                except Exception as e:
                    import streamlit as st
                    st.write(f"UnstructuredExcelLoader加载失败: {e}，尝试使用pandas")
            
            # 备用方案：使用pandas读取Excel文件
            try:
                # 读取Excel文件的所有工作表
                excel_file = pd.ExcelFile(file_path)
                documents = []
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # 将DataFrame转换为文本
                    text_content = f"工作表: {sheet_name}\n\n"
                    
                    # 添加列名
                    if not df.empty:
                        text_content += "列名: " + ", ".join(df.columns.astype(str)) + "\n\n"
                        
                        # 添加数据行
                        for index, row in df.iterrows():
                            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                            if row_text.strip():
                                text_content += f"行{index + 1}: {row_text}\n"
                    
                    # 创建文档对象
                    from langchain.schema import Document
                    doc = Document(
                        page_content=text_content,
                        metadata={"source": file_path, "sheet_name": sheet_name}
                    )
                    documents.append(doc)
                
                return documents
                
            except Exception as e:
                import streamlit as st
                st.write(f"pandas加载Excel文件失败: {e}")
                st.write(f"文件路径: {file_path}")
                st.write(f"文件类型: {file_type}")
                return None
        else:
            return None
        
        # 对于非Excel文件，加载文档
        documents = loader.load()
        return documents
    except Exception as e:
        import streamlit as st
        st.write(f"加载文件失败: {e}")
        st.write(f"文件路径: {file_path}")
        st.write(f"文件类型: {file_type}")
        return None

### 2. 文本分割器

def split_documents(documents):
    """将文档分割成小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每个块的大小
        chunk_overlap=100,   # 块之间的重叠
        length_function=len,
    )
    
    splits = text_splitter.split_documents(documents)
    return splits

### 3. 文档嵌入和存储

def generate_embeddings(texts, model):
    """生成文本的嵌入向量
    
    Args:
        texts: 文本列表
        model: 嵌入模型
    
    Returns:
        list: 嵌入向量列表
    """
    try:
        embeddings = model.encode(texts).tolist()
        return embeddings
    except Exception as e:
        import streamlit as st
        st.write(f"生成嵌入向量失败: {e}")
        return None

def store_documents_to_collection(texts, embeddings, metadatas, ids, collection):
    """将文档存储到Chroma集合
    
    Args:
        texts: 文档文本列表
        embeddings: 嵌入向量列表
        metadatas: 元数据列表
        ids: 文档ID列表
        collection: Chroma集合对象
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return True, f"成功存储 {len(texts)} 个文档"
    except Exception as e:
        return False, f"存储失败: {str(e)}"

def delete_documents_by_filename(file_name, collection):
    """根据文件名删除向量数据库中的相关记录
    
    Args:
        file_name: 要删除的文件名
        collection: Chroma集合对象
    
    Returns:
        tuple: (success: bool, deleted_count: int, message: str)
    """
    try:
        # 1. 查询所有记录
        all_results = collection.get(include=['metadatas'])
        
        # 2. 找到匹配文件名的记录ID
        ids_to_delete = []
        for i, metadata in enumerate(all_results['metadatas']):
            if metadata and metadata.get('file_name') == file_name:
                ids_to_delete.append(all_results['ids'][i])
        
        # 3. 删除匹配的记录
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return True, len(ids_to_delete), f"成功删除 {len(ids_to_delete)} 条向量记录"
        else:
            return True, 0, f"未找到文件 {file_name} 的向量记录"
            
    except Exception as e:
        return False, 0, f"删除向量记录失败: {str(e)}"

def get_documents_by_filename(file_name, collection):
    """根据文件名获取所有相关的向量记录
    
    Args:
        file_name: 文件名
        collection: Chroma集合对象
    
    Returns:
        list: 该文件的所有向量记录
    """
    try:
        # 查询所有记录
        all_results = collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        # 筛选匹配文件名的记录
        file_records = []
        for i, metadata in enumerate(all_results['metadatas']):
            if metadata and metadata.get('file_name') == file_name:
                file_records.append({
                    "id": all_results['ids'][i],
                    "document": all_results['documents'][i],
                    "metadata": metadata,
                    "embedding": all_results['embeddings'][i] if all_results['embeddings'] else None
                })
        
        return file_records
        
    except Exception as e:
        import streamlit as st
        st.write(f"获取文件记录失败: {e}")
        return []

def get_file_statistics(collection):
    """获取所有文件的统计信息
    
    Args:
        collection: Chroma集合对象
    
    Returns:
        dict: 文件统计信息
    """
    try:
        # 查询所有记录
        all_results = collection.get(include=['metadatas'])
        
        # 统计每个文件的信息
        file_stats = {}
        for metadata in all_results['metadatas']:
            if metadata:
                file_name = metadata.get('file_name', '未知文件')
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        "file_name": file_name,
                        "file_type": metadata.get('file_type', '未知'),
                        "total_chunks": metadata.get('total_chunks', 0),
                        "chunk_count": 0,
                        "file_path": metadata.get('file_path', '未知路径')
                    }
                file_stats[file_name]["chunk_count"] += 1
        
        return file_stats
        
    except Exception as e:
        import streamlit as st
        st.write(f"获取文件统计失败: {e}")
        return {}

def search_documents(query, collection, model, n_results=5, file_filter=None):
    """在数据库中搜索相似文档
    
    Args:
        query: 查询文本
        collection: Chroma集合对象
        model: 嵌入模型
        n_results: 返回结果数量
        file_filter: 文件名过滤（可选）
    
    Returns:
        list: 搜索结果列表（按相似度排序）
    """
    try:
        # 1. 将查询转换为嵌入向量
        query_embedding = model.encode([query]).tolist()
        
        # 2. 构建查询条件
        where_condition = None
        if file_filter:
            where_condition = {"file_name": file_filter}
        
        # 3. 在数据库中搜索
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
            where=where_condition
        )
        
        # 4. 处理结果
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # 将距离转换为相似度
                similarity = 1 - distance
                
                search_results.append({
                    "文档": metadata['file_name'],
                    "相似度": round(similarity, 3),
                    "内容": doc,
                    "文件类型": metadata['file_type'],
                    "块索引": metadata['chunk_index'],
                    "总块数": metadata.get('total_chunks', '未知')
                })
        
        return search_results
        
    except Exception as e:
        import streamlit as st
        st.write(f"搜索失败: {e}")
        return []
