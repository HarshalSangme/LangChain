from langchain_community.document_loaders import text
from langchain.text_splitter import RecursiveCharacterTextSplitter

my_text = text.TextLoader(file_path='sample_text.txt').load()

sentence_splitter = RecursiveCharacterTextSplitter(
    separators=[".", "?", "!"],
    chunk_size=100,
    chunk_overlap=0
)

# Split the documents by passing a list of Document objects
text_split = sentence_splitter.split_documents(documents=[my_text[0]])

# Print the resulting split text
for idx, chunk in enumerate(text_split):
    print(f"Sentence {idx + 1}: {chunk.page_content.strip()}")
