import os
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            ['answer[:2000]' [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF document, split it into chunks, and create a retriever for question answering.

        :param pdf_file_path: Path to the PDF document to be ingested.
        """
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            if not docs:
                return "Failed to load the PDF document."
            
            print(f"Loaded {len(docs)} documents from PDF.")
            print("Full document content length:", sum(len(doc.page_content) for doc in docs))

            # for i, doc in enumerate(docs):
            #     print(f"Document {i} content preview:", doc.page_content[:500])  # Preview first 500 chars

            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                return "Failed to split the document into chunks."

            print(f"Split into {len(chunks)} chunks.")
            # print("First chunk preview:", chunks[0].page_content[:500])  # Preview first 500 chars

            chunks = filter_complex_metadata(chunks)
            print(f"Filtered chunks: {len(chunks)}")

            self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
            if not self.vector_store:
                return "Failed to create vector store."

            print("Vector store created successfully.")

            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.2,
                },
            )
            if not self.retriever:
                return "Failed to create retriever."

            print("Retriever created successfully.")

            self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                          | self.prompt
                          | self.model
                          | StrOutputParser())
            print("Chain created successfully.")
        except Exception as e:
            return f"Error during ingestion: {str(e)}"


    def ask(self, query: str) -> str:
        """
        Ask a question based on the ingested PDF content.

        :param query: The question to ask.
        :return: The answer to the question.
        """
        if not self.chain:
            return "Please, add a PDF document first."
        try:
            return self.chain.invoke(query)
        except Exception as e:
            return f"Error during query processing: {str(e)}"

    def clear(self):
        """
        Clear the stored vector store, chain and retriever.
        """
        self.vector_store = None
        self.retriever = None
        self.chain = None


# Test the ChatPDF class
# chat_pdf = ChatPDF()

# # Path to a sample PDF file
# pdf_path = "18140-Final PDF-23063-1-10-20220111.pdf"  # Replace this with the path to your PDF file

# # Ingest the PDF file
# ingest_result = chat_pdf.ingest(pdf_path)
# if ingest_result is not None:
#     print(ingest_result)
# else:
#     # Ask a question
#     question = "Give me a complete sumarry of the Introduction on the paper"
#     answer = chat_pdf.ask(question)
#     print("Answer:", answer)
