from langchain_community.document_loaders import text
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

docs = text.TextLoader('ai_text.txt')
my_docs = docs.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  
    chunk_overlap=50 
)

page_content = []
for doc in my_docs:
    chunks = text_splitter.split_text(doc.page_content)
    page_content.extend(chunks)

print(f"Number of chunks: {len(page_content)}")
print(f"First few chunks: {page_content[:3]}")

embedding_model = GPT4AllEmbeddings()
generated_embeddings = embedding_model.embed_documents(page_content)

d = len(generated_embeddings[0])

index = faiss.IndexFlatL2(d)

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

metadata = [{"id": i, "content": doc} for i, doc in enumerate(page_content)]
text_embeddings = list(zip(page_content, generated_embeddings))

with open('new_embeddings.txt', 'w') as embedding_file:
    embedding_file.write(str(text_embeddings)) 

vector_store.add_embeddings(text_embeddings)

vector_store.save_local('faiss_index')

print(f"Added {len(generated_embeddings)} embeddings to FAISS and saved index.")

search_result = vector_store.similarity_search('Machine perception', k=2)

print("search_result:", search_result)
