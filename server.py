from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document

def embedd_text():
    text_data = TextLoader("sample_text.txt").load()

    embeddings = GPT4AllEmbeddings().embed_documents(
        texts=[text_data[0].page_content])

    documents = []
    doc_ids = []
    
    for idx, embedding in enumerate(embeddings):
        doc_format = Document(page_content=text_data[idx].page_content)
        documents.append(doc_format)
        doc_ids.append(idx)        
    
    index = FAISS.add_documents(FAISS, documents)
    vector_store = FAISS(embedding_function=GPT4AllEmbeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    print(vector_store)

embedd_text()
