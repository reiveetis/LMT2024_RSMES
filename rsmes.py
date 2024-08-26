import ollama
from ollama import AsyncClient
import chromadb
from semantic_text_splitter import TextSplitter
from pypdf import PdfReader
import os
import time
import asyncio
from flashrank import Ranker, RerankRequest

EMBEDDING_MODEL = 'mxbai-embed-large'
LLM = 'internlm2'
DATA_PATH = './data/'
KEY_PATH = '.api-key'
DB_PATH = 'chroma'
COLLECTION_NAME = 'test'


def read_pdf_file(path, filename):
    result = []
    if filename.endswith('.pdf'):
        filepath = os.path.join(path, filename)
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            page_count = reader.get_num_pages()
            for curr_page in range(page_count):
                print(f'Reading {filename}... {curr_page + 1}/{page_count}')
                page = reader.pages[curr_page]
                text = page.extract_text()
                result.append(
                    {
                        'content':text,
                        'metadata':{
                            'source':filename,
                            'page':curr_page + 1
                        }
                    }
                )
    return result

def read_unique_pdf_files_in_dir(path, collection):
    result = []
    db_sources = get_sources_in_collection(collection)
    files = os.listdir(path)
    for file in files:
        if file.endswith('pdf'):
            if file not in db_sources:
                response = read_pdf_file(path, file)
                for page in response:
                    result.append(page)
    return result

def split_documents(docs):
    result = []
    for doc in docs:
        splitter = TextSplitter(800, 60)
        chunks = splitter.chunks(doc['content'])
        for chunk in chunks:
            result.append(
            {
                'content':chunk,
                'metadata':doc['metadata']
            }
        )
    return result

def generate_chunk_ids(chunks):
    result = []
    prev_page = 1
    curr_id = 0
    for item in chunks:
        page = item['metadata']['page']
        source = item['metadata']['source']
        if (prev_page == page):
            curr_id += 1
        else:
            prev_page = page
            curr_id = 0
        id = f'{source}:{page}:{curr_id}'
        result.append(
            {
                'content':item['content'],
                'metadata':{
                    'source':source,
                    'page':page,
                    'id':id
                }
            }
        )
    return result
        
def get_sources_in_collection(collection):
    # 'metadatas' is empty, therefore this only returns ids
    ids = collection.get(include=['metadatas'])['ids']
    db_sources = []

    for id in ids:
        source = id.split(':')[0]
        if source not in db_sources:
            db_sources.append(source)
    return db_sources
        
def build_collection(docs, collection):
    split = split_documents(docs)
    chunks = generate_chunk_ids(split)
    total_chunks = len(chunks)
    curr_chunk = 0
    for chunk in chunks:
        curr_chunk += 1
        print(f'Embedding chunks... {curr_chunk}/{total_chunks}')
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk['content'])
        id = chunk['metadata']['id']
        document = chunk['content']
        embedding = response['embedding']
        if id is not None:
            collection.add(
                documents=[document],
                ids=[id],
                embeddings=[embedding]
            )

def print_help():
    print("Usage:")
    print(f"1. Add .pdf files to '{DATA_PATH}'")
    print("2. Build embeddings with '/build'")
    print("3. Enter a prompt")
    print("Commands:")
    print(f"/build - builds embeddings from .pdf files in '{DATA_PATH}'")
    print("/quit - quits the program")
    print("/context - prints the context given to the llm with the previous prompt")

def generate_answer(query, size, top_result_count, collection, should_rerank = False):
    result = generate_context_with_sources(query, size, collection)
    formatted = result[0]
    sources = result[1]
    
    if should_rerank:
        ranker = Ranker()
        rerankrequest = RerankRequest(query, formatted)
        reranked = ranker.rerank(rerankrequest)
    else:
        reranked = formatted
    
    context = ''
    for item in reranked:
        context += item['text']
        context += '\n\n---\n\n'
    message_system = {
        'role':'system',
        'content':f"Answer the user's question based on this context:\n\nContext:\n{context}"
    }
    message_user = {
        'role':'user',
        'content':query
    }
    response = ollama.chat(model=LLM, messages=[message_system, message_user])
    print(f'Source count: {len(sources)}')
    print(context)
    return response['message']['content']

def generate_context_with_sources(query, size, collection):
    print('Generating query embeddings...')
    response = ollama.embeddings(
        prompt=query,
        model=EMBEDDING_MODEL,
    )
    
    print('Getting results from db...')
    results = collection.query(
        query_embeddings=[response['embedding']],
        n_results=size
    )
    
    formatted = []
    for i, _ in enumerate(results['documents'][0]):
        formatted.append(
            {'text': results['documents'][0][i]}
        )

    sources = [DATA_PATH + id for id in results['ids'][0]]
    return [formatted, sources]

async def generate_answer_stream(query, context_size, collection):
    result = generate_context_with_sources(query, context_size, collection)
    formatted = result[0]
    sources = result[1]
    
    print('Reranking context...')
    ranker = Ranker()
    rerankrequest = RerankRequest(query, formatted)
    reranked = ranker.rerank(rerankrequest)

    context = ''
    for item in reranked:
        context += item['text']
        context += '\n\n---\n\n'

    message_system = {
        'role':'system',
        'content':f"Answer the user's question based on this context:\n\nContext:\n{context}"
    }
    message_user = {
        'role':'user',
        'content':query
    }
    
    print('Generating answer...')
    async for part in await AsyncClient().chat(model=LLM, messages=[message_system, message_user], stream=True, options={'num_ctx': 9830}):
        print(part['message']['content'], end='', flush=True)
    print('\nSources: ' + str(sources))
    return context


db = chromadb.PersistentClient(DB_PATH)
collection = db.get_or_create_collection(COLLECTION_NAME)
context = ''

if __name__ == '__main__':
    while True:
        print("Enter prompt or type '/help' for the list of commands")
        user_prompt = input(">> ")
        if (user_prompt.lower() == '/help'):
            print_help()
        elif (user_prompt.lower() == '/build'):
            start_time = time.time()
            docs = read_unique_pdf_files_in_dir(DATA_PATH, collection)
            build_collection(docs, collection)
            print(f'Building took {time.time() - start_time} seconds.')
        elif (user_prompt.lower() == '/quit'):
            quit()
        elif (user_prompt.lower() == '/context'):
            if (context == ''):
                print('Empty context!')
            else:
                print(context)
        elif (user_prompt[0] == '/'):
            print('Unknown command!')
        else:
            start_time = time.time()
            context = asyncio.run(generate_answer_stream(user_prompt, 18, collection))
            print(f'Generating the answer took {time.time() - start_time} seconds.')