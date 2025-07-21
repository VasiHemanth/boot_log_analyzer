#!/usr/bin/env python3
"""
Direct File Analysis using LangChain with Ollama
Simple analysis without vector stores - just load and analyze
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader
)
from langchain.prompts import PromptTemplate

class SimpleFileAnalyzer:
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize the analyzer with Ollama model
        
        Args:
            model_name: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
        """
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Larger chunks since we're not using vector store
            chunk_overlap=200,
            length_function=len
        )
        self.documents = []
        self.file_content = ""

    def load_file(self, file_path: str) -> str:
        """
        Load and extract text from various file types
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            # Choose appropriate loader
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.txt' or file_extension == '.log':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_extension == '.csv':
                loader = CSVLoader(str(file_path))
            elif file_extension == '.json':
                loader = JSONLoader(str(file_path), jq_schema='.')
            else:
                loader = TextLoader(str(file_path), encoding='utf-8')
            
            # Load documents
            documents = loader.load()
            print(f"Loaded {len(documents)} document(s) from {file_path.name}")
            
            # Extract all text content
            self.file_content = "\n\n".join([doc.page_content for doc in documents])
            self.documents = documents
            
            print(f"Extracted {len(self.file_content)} characters")
            return self.file_content
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return ""

    def summarize_content(self, max_length: int = 500) -> str:
        """
        Create a summary of the file content
        
        Args:
            max_length: Maximum length of summary in words
            
        Returns:
            Content summary
        """
        if not self.file_content:
            return "No content loaded."
        
        prompt_template = """
        Summarize the following document content in approximately {max_length} words or less.
        Focus on the main topics, key points, and overall purpose of the document.

        Document Content:
        {content}

        Summary:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content", "max_length"]
        )
        
        chain = prompt | self.llm         

        # Truncate content if too long
        content = self.file_content[:8000] if len(self.file_content) > 8000 else self.file_content
        
        response = chain.invoke({"content": content, "max_length": max_length})

        return response

    def extract_key_points(self) -> str:
        """
        Extract key points and important information
        
        Returns:
            Key points from the document
        """
        if not self.file_content:
            return "No content loaded."
        
        prompt_template = """
        Analyze the following document and extract:
        1. Main topics and themes
        2. Key facts and figures
        3. Important names, dates, and locations
        4. Significant conclusions or recommendations
        5. Any action items or next steps

        Present the information in a clear, organized format.

        Document Content:
        {content}

        Key Points:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"]
        )
        
        chain = prompt | self.llm
        
        # Use first 8000 characters to avoid token limits
        content = self.file_content[:8000] if len(self.file_content) > 8000 else self.file_content
        
        response = chain.invoke({"content": content})
        return response

    def analyze_sentiment(self) -> str:
        """
        Analyze the sentiment and tone of the document
        
        Returns:
            Sentiment analysis
        """
        if not self.file_content:
            return "No content loaded."
        
        prompt_template = """
        Analyze the sentiment and tone of the following document:
        - Overall sentiment (positive, negative, neutral)
        - Emotional tone
        - Author's attitude
        - Level of formality
        - Any bias detected

        Document Content:
        {content}

        Sentiment Analysis:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"]
        )
        
        chain = prompt | self.llm
        
        content = self.file_content[:6000] if len(self.file_content) > 6000 else self.file_content
        
        response = chain.invoke({"content": content})
        return response

    def classify_document(self) -> str:
        """
        Classify the type and category of the document
        
        Returns:
            Document classification
        """
        if not self.file_content:
            return "No content loaded."
        
        prompt_template = """
        Classify the following document by:
        1. Document type (report, article, email, manual, etc.)
        2. Subject domain (business, technical, academic, legal, etc.)
        3. Purpose (informational, instructional, persuasive, etc.)
        4. Target audience
        5. Level of complexity

        Document Content:
        {content}

        Classification:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["content"]
        )
        
        chain = prompt | self.llm
        
        content = self.file_content[:5000] if len(self.file_content) > 5000 else self.file_content
        
        response = chain.invoke({"content": content})
        return response

    def custom_analysis(self, analysis_prompt: str) -> str:
        """
        Perform custom analysis based on user-provided prompt
        
        Args:
            analysis_prompt: Custom analysis instructions
            
        Returns:
            Analysis based on custom prompt
        """
        if not self.file_content:
            return "No content loaded."
        
        prompt_template = """
        {analysis_instruction}

        Document Content:
        {content}

        Analysis:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["analysis_instruction", "content"]
        )
        
        chain = prompt | self.llm
        
        content = self.file_content[:8000] if len(self.file_content) > 8000 else self.file_content
        
        response = chain.invoke({"analysis_instruction": analysis_prompt, "content": content})
        return response

    def get_file_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the loaded file
        
        Returns:
            Dictionary with file statistics
        """
        if not self.file_content:
            return {"error": "No content loaded"}
        
        words = len(self.file_content.split())
        characters = len(self.file_content)
        lines = len(self.file_content.split('\n'))
        
        return {
            "characters": characters,
            "words": words,
            "lines": lines,
            "documents": len(self.documents)
        }

def analyze_file(file_path: str, model_name: str = "llama2") -> Dict[str, str]:
    """
    Simple function to analyze a file and return all analyses
    
    Args:
        file_path: Path to the file to analyze
        model_name: Ollama model to use
        
    Returns:
        Dictionary with all analysis results
    """
    analyzer = SimpleFileAnalyzer(model_name=model_name)
    
    try:
        # Load file
        content = analyzer.load_file(file_path)
        
        if not content:
            return {"error": "Failed to load file or file is empty"}
        
        results = {}
        
        # Get file statistics
        results["file_stats"] = analyzer.get_file_stats()
        
        # Perform different types of analysis
        print("Generating summary...")
        results["summary"] = analyzer.summarize_content()
        
        print("Extracting key points...")
        results["key_points"] = analyzer.extract_key_points()
        
        print("Analyzing sentiment...")
        results["sentiment"] = analyzer.analyze_sentiment()
        
        print("Classifying document...")
        results["classification"] = analyzer.classify_document()
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Example usage of the SimpleFileAnalyzer"""
    
    # Get file path from user
    # file_path = input("Enter the path to your file: ").strip()
    # file_path="/Users/hemanthvasi/Documents/Developer/Gen AI Projects/sai/boot.log"
    file_path="/Users/hemanthvasi/Documents/Developer/Gen AI Projects/uefi_log_analyzer/boot.log"
    
    if not file_path:
        print("No file path provided. Using example...")
        file_path = "example.txt"  # Replace with your test file
    
    # Initialize analyzer
    analyzer = SimpleFileAnalyzer(model_name="gemma3:latest")  # Change model as needed
    
    try:
        print(f"\nAnalyzing file: {file_path}")
        print("=" * 60)
        
        # Load file
        content = analyzer.load_file(file_path)
        
        if not content:
            print("Failed to load file or file is empty.")
            return
        
        # Show file stats
        stats = analyzer.get_file_stats()
        print(f"\nFile Statistics:")
        print(f"- Characters: {stats['characters']:,}")
        print(f"- Words: {stats['words']:,}")
        print(f"- Lines: {stats['lines']:,}")
        
        # Menu for different analyses
        while True:
            print("\n" + "=" * 60)
            print("Choose analysis type:")
            print("1. Summary")
            print("2. Key Points")
            print("3. Sentiment Analysis")
            print("4. Document Classification")
            print("5. Custom Analysis")
            print("6. All Analyses")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                print("\nSUMMARY:")
                print("-" * 40)
                print(analyzer.summarize_content())
            elif choice == "2":
                print("\nKEY POINTS:")
                print("-" * 40)
                print(analyzer.extract_key_points())
            elif choice == "3":
                print("\nSENTIMENT ANALYSIS:")
                print("-" * 40)
                print(analyzer.analyze_sentiment())
            elif choice == "4":
                print("\nDOCUMENT CLASSIFICATION:")
                print("-" * 40)
                print(analyzer.classify_document())
            elif choice == "5":
                custom_prompt = input("Enter your custom analysis prompt: ")
                print("\nCUSTOM ANALYSIS:")
                print("-" * 40)
                print(analyzer.custom_analysis(custom_prompt))
            elif choice == "6":
                # Run all analyses
                analyses = analyze_file(file_path, model_name="gemma3:latest")  # Change model as needed
                for key, value in analyses.items():
                    if key != "file_stats":
                        print(f"\n{key.upper().replace('_', ' ')}:")
                        print("-" * 40)
                        print(value)
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()