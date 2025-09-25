import streamlit as st
import pandas as pd
import os
from datetime import datetime
import uuid
import chroma

# 初始化Chroma数据库和嵌入模型
collection = chroma.init_chroma_db()
model = chroma.init_embedding_model()

# 页面配置
st.set_page_config(
    page_title="知识库管理系统",
    page_icon="📚",
    layout="wide"
)

# 初始化session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# 创建保存目录
save_dir = "知识库文件"
os.makedirs(save_dir, exist_ok=True)

# 标题
st.title("📚 知识库管理系统")
st.markdown("---")

# 功能选择
gongneng = ["上传知识库", "知识库查询", "删除知识库"]
# horizontal=True是水平排列
selected_function = st.radio("选择功能:", gongneng, horizontal=True)

st.markdown("---")

# 1. 上传知识库功能
if selected_function == "上传知识库":
    st.header("📤 上传知识库")
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "选择要上传的文档:",
        type=['txt', 'pdf', 'docx', 'md', 'xlsx'],
        accept_multiple_files=True,
        help="支持上传多个文件"
    )
    
    if uploaded_files:
        st.success(f"成功选择 {len(uploaded_files)} 个文件")
        
        # 显示文件信息
        file_info = []
        for file in uploaded_files:
            file_info.append({
                "文件名": file.name,
                "文件大小": f"{file.size / 1024:.2f} KB",
                "文件类型": file.type
            })
        
        # 用pandas库创建一个数据框
        # 用dataframe函数显示数据框
        df = pd.DataFrame(file_info)
        # use_container_width=True是使用容器宽度
        st.dataframe(df, use_container_width=True)
        
        # 上传按钮
        if st.button("🚀 开始上传", type="primary"):
            # with 就是在程序开始到结束的中间进行显示效果
            with st.spinner("正在上传文件..."):
                progress_bar = st.progress(0)
                
                # 真正保存文件到磁盘
                for i, file in enumerate(uploaded_files):
                    try:
                        # 构建文件保存路径
                        file_path = os.path.join(save_dir, file.name)
                        print(file_path)
                        # 保存文件到磁盘
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # 保存文件信息到session state
                        st.session_state.uploaded_files.append({
                            "name": file.name,
                            "path": file_path,  # 添加文件路径
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # 更新进度条
                        progress = int((i + 1) / len(uploaded_files) * 100)
                        progress_bar.progress(progress)

                        # 1. 加载文档
                        documents = chroma.load_document(file_path, file.type)
                        if not documents:
                            st.warning(f"⚠️ 无法加载文件: {file.name}")
                            continue
                        
                        # 2. 分割文档
                        splits = chroma.split_documents(documents)
                        if not splits:
                            st.warning(f"⚠️ 无法分割文件: {file.name}")
                            continue
                        
                        # 3. 准备数据
                        texts = [split.page_content for split in splits]
                        metadatas = []
                        ids = []
                        
                        for i, split in enumerate(splits):
                            metadata = {
                                "file_name": file.name,
                                "file_type": file.type,
                                "file_path": file_path,
                                "chunk_index": i,
                                "total_chunks": len(splits),
                                "embedding_type": "knowledge_base"
                            }
                            metadatas.append(metadata)
                            ids.append(f"kb_{file.name}_{i}_{uuid.uuid4().hex[:8]}")
                        
                        # 4. 生成嵌入向量
                        embeddings = chroma.generate_embeddings(texts, model)
                        if embeddings is None:
                            st.error(f"❌ 生成嵌入向量失败: {file.name}")
                            continue
                        
                        # 5. 存储到数据库
                        success, message = chroma.store_documents_to_collection(texts, embeddings, metadatas, ids, collection)
                        if success:
                            st.success(f"✅ {file.name}: {message}")
                        else:
                            st.error(f"❌ {file.name}: {message}")
                        
                    except Exception as e:
                        st.error(f"保存文件 {file.name} 时出错: {str(e)}")
                        continue
            
            st.success(f"✅ 成功保存 {len(uploaded_files)} 个文件到 {save_dir} 文件夹！")
            st.balloons()

# 2. 知识库查询功能
elif selected_function == "知识库查询":
    st.header("🔍 知识库查询")
    
    # 查询输入
    query = st.text_area(
        "输入查询内容:",
        placeholder="例如：项目的主要功能是什么？",
        height=100,
        key="query_input"
    )
    
    # 查询选项
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.number_input("最大结果数", 1, 20, 5)
    
    with col2:
        # 获取所有文件名作为过滤选项
        try:
            all_results = collection.get(include=['metadatas'])
            file_names = set()
            for metadata in all_results['metadatas']:
                if metadata and metadata.get('file_name'):
                    file_names.add(metadata['file_name'])
            
            file_names = sorted(list(file_names))
            if file_names:
                file_filter = st.selectbox("按文件过滤", ["全部文件"] + file_names)
                file_filter = None if file_filter == "全部文件" else file_filter
            else:
                file_filter = None
                st.info("暂无文件可过滤")
        except:
            file_filter = None
    
    # 查询按钮
    if st.button("🔍 开始查询", type="primary"):
        if query:
            with st.spinner("正在查询..."):
                # 执行向量搜索
                results = chroma.search_documents(query, collection, model, n_results=max_results, file_filter=file_filter)
                
                if results:
                    filter_text = f" (在 {file_filter} 中)" if file_filter else ""
                    st.success(f"找到 {len(results)} 个相关结果{filter_text}")
                    
                    # 显示结果
                    for i, result in enumerate(results, 1):
                        with st.expander(f"结果 {i}: {result['文档']} (相似度: {result['相似度']:.2f}) - 块 {result['块索引']+1}/{result['总块数']}"):
                            st.write(result["内容"])
                else:
                    filter_text = f"在 {file_filter} 中" if file_filter else ""
                    st.warning(f"未找到相关结果{filter_text}，请尝试调整查询条件")
        else:
            st.warning("请输入查询内容")

# 3. 删除知识库功能
elif selected_function == "删除知识库":
    st.header("🗑️ 删除知识库")
    
    # 直接扫描文件夹中的所有文件
    if os.path.exists(save_dir):
        files_in_dir = os.listdir(save_dir)
        
        if files_in_dir:
            st.write(f"📁 文件夹中的所有文件 ({len(files_in_dir)} 个):")
            
            # 创建选择框
            selected_files = []
            for i, file_name in enumerate(files_in_dir):
                file_path = os.path.join(save_dir, file_name)
                file_size = os.path.getsize(file_path)
                
                if st.checkbox(f"🗂️ {file_name} ({file_size/1024:.2f} KB)", key=f"file_{i}"):
                    selected_files.append(file_name)
            
            if selected_files:
                st.warning(f"已选择 {len(selected_files)} 个文件")
                
                # 显示将要删除的文件
                st.subheader("将要删除的文件:")
                for file_name in selected_files:
                    st.write(f"• {file_name}")
                
                # 确认删除
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ 确认删除", type="primary"):
                        deleted_count = 0
                        error_count = 0
                        
                        for file_name in selected_files:
                            file_path = os.path.join(save_dir, file_name)
                            try:
                                # 1. 删除向量数据库中的记录
                                vector_success, vector_count, vector_message = chroma.delete_documents_by_filename(file_name, collection)
                                
                                # 2. 删除文件系统中的文件
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    deleted_count += 1
                                    
                                    # 显示删除结果
                                    if vector_success and vector_count > 0:
                                        st.success(f"✅ 删除成功: {file_name} (文件 + {vector_count} 条向量记录)")
                                    elif vector_success and vector_count == 0:
                                        st.success(f"✅ 删除成功: {file_name} (文件，未找到向量记录)")
                                    else:
                                        st.warning(f"⚠️ 文件删除成功: {file_name}，但向量删除失败: {vector_message}")
                                else:
                                    st.warning(f"⚠️ 文件不存在: {file_name}")
                                    if vector_success and vector_count > 0:
                                        st.info(f"ℹ️ 已删除 {vector_count} 条向量记录")
                                        
                            except Exception as e:
                                st.error(f"❌ 删除失败 {file_name}: {str(e)}")
                                error_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"🎉 总共删除了 {deleted_count} 个文件！")
                        if error_count > 0:
                            st.warning(f"⚠️ {error_count} 个文件删除失败")
                        
                        st.rerun()
                
                with col2:
                    # 预留位置，可以添加其他功能
                    pass
            else:
                st.info("请选择要删除的文件")
        else:
            st.info("📁 知识库文件夹为空")
    else:
        st.info("📂 知识库文件夹不存在")
    
    # 显示保存目录信息
    st.info(f"📂 文件保存位置: {os.path.abspath(save_dir)}")

# 侧边栏 - 系统状态
with st.sidebar:
    st.header("📊 系统状态")
    
    # 检查向量数据库存储状态
    if st.button("🔍 检查向量数据库", type="secondary"):
        try:
            # 查询数据库中的所有数据
            results = collection.get()
            
            if results['ids']:
                st.success(f"✅ 向量数据库存储成功！共有 {len(results['ids'])} 条向量记录")
                
                # 显示存储统计
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总记录数", len(results['ids']))
                with col2:
                    # 统计知识库嵌入数量
                    kb_count = 0
                    for metadata in results['metadatas']:
                        if metadata and metadata.get('embedding_type') == 'knowledge_base':
                            kb_count += 1
                    st.metric("知识库嵌入", kb_count)
                
                # 显示文件统计信息
                st.subheader("📊 文件统计")
                file_stats = chroma.get_file_statistics(collection)
                if file_stats:
                    for file_name, stats in file_stats.items():
                        with st.expander(f"📄 {file_name} ({stats['chunk_count']}/{stats['total_chunks']} 块)"):
                            st.write(f"**文件类型:** {stats['file_type']}")
                            st.write(f"**实际块数:** {stats['chunk_count']}")
                            st.write(f"**预期块数:** {stats['total_chunks']}")
                            st.write(f"**文件路径:** {stats['file_path']}")
                
                # 显示最近3条记录
                st.write("**最近存储的记录:**")
                for i in range(min(3, len(results['ids']))):
                    with st.expander(f"📄 {results['metadatas'][i].get('file_name', '未知文件')} - 块 {i+1}"):
                        st.write("**ID:**", results['ids'][i])
                        st.write("**文件路径:**", results['metadatas'][i].get('file_path', '未知'))
                        st.write("**文本预览:**", results['documents'][i][:150] + "...")
                        st.write("**向量维度:**", len(results['embeddings'][i]) if results['embeddings'] else "无")
            else:
                st.warning("⚠️ 向量数据库为空，还没有存储任何向量")
                
        except Exception as e:
            st.error(f"❌ 检查向量数据库失败: {e}")
    
    # 文件统计 - 直接读取路径里的文件
    if os.path.exists(save_dir):
        files_in_dir = os.listdir(save_dir)
        file_count = len(files_in_dir)
        st.metric("已有文件", file_count)
        
        # 文件大小统计
        total_size = 0
        for file_name in files_in_dir:
            file_path = os.path.join(save_dir, file_name)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        st.metric("总大小", f"{total_size/1024/1024:.2f} MB")
    else:
        st.metric("已有文件", 0)
        st.metric("总大小", "0 MB")
    
    # 保存目录信息
    st.subheader("📂 保存位置")
    st.text(os.path.abspath(save_dir))
    
    # 清空数据按钮
    if st.button("🗑️ 清空所有数据"):
        # 1. 清空向量数据库
        try:
            # 获取所有记录
            all_results = collection.get()
            if all_results['ids']:
                # 删除所有向量记录
                collection.delete(ids=all_results['ids'])
                st.success(f"✅ 已清空向量数据库 ({len(all_results['ids'])} 条记录)")
            else:
                st.info("ℹ️ 向量数据库已为空")
        except Exception as e:
            st.error(f"❌ 清空向量数据库失败: {str(e)}")
        
        # 2. 清空文件夹
        deleted_count = 0
        if os.path.exists(save_dir):
            files_in_dir = os.listdir(save_dir)
            for file_name in files_in_dir:
                file_path = os.path.join(save_dir, file_name)
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    st.error(f"删除文件 {file_name} 失败: {str(e)}")
        
        # 3. 清空session state
        st.session_state.uploaded_files = []
        
        st.success(f"🎉 数据已完全清空！删除了 {deleted_count} 个文件")
        st.rerun()