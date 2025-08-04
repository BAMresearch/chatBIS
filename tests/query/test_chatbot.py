#!/usr/bin/env python3
"""
Test script to test the chatbot with a specific query.
"""

import json
from chatBIS.query.query import RAGQueryEngine

def main():
    # Initialize the query engine
    query_engine = RAGQueryEngine(
        data_dir="data/processed",
        model="qwen3"
    )
    
    # Test query
    query = "hey! I am a new openBIs user, I will need you to help me!"
    
    # First, let's check if we have the "Register a Collection" chunk
    register_collection_chunks = []
    for chunk in query_engine.chunks:
        if "register a collection" in chunk["content"].lower():
            register_collection_chunks.append(chunk)
    
    if register_collection_chunks:
        print(f"Found {len(register_collection_chunks)} 'Register a Collection' chunks:")
        for i, chunk in enumerate(register_collection_chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Title: {chunk['title']}")
            print(f"URL: {chunk['url']}")
            print(f"Content: {chunk['content']}")
        print("\n" + "-" * 80 + "\n")
    else:
        print("Could not find 'Register a Collection' chunk")
        print("\n" + "-" * 80 + "\n")
    
    # Generate answer
    answer, relevant_chunks = query_engine.query(query)
    
    # Print the answer
    print(f"Query: {query}")
    print(f"\nAnswer:\n{answer}")
    
    # Print the relevant chunks
    print(f"\nRelevant chunks:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Content: {chunk['content'][:200]}...")

if __name__ == "__main__":
    main()
