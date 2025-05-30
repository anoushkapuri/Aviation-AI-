# Aviation-AI-import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_engine import QueryEngine
from utils import get_document_metadata, save_document_metadata, delete_document_metadata

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = QueryEngine(st.session_state.vector_store)
if 'document_processor' not in st.session_state:
    st.session_state.document_processor = DocumentProcessor()
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

def main():
    st.title("Aviation AI Assistant")
    st.markdown("Upload aviation PDF documents and query them using natural language")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Aviation PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload aviation standards, manuals, or planning documents"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if already processed
                if uploaded_file.name not in [doc['name'] for doc in st.session_state.uploaded_documents]:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                tmp_file_path = tmp_file.name
                            
                            # Process document
                            chunks = st.session_state.document_processor.process_pdf(
                                tmp_file_path, uploaded_file.name
                            )
                            
                            if chunks:
                                # Add to vector store
                                st.session_state.vector_store.add_documents(chunks)
                                
                                # Save metadata
                                doc_metadata = {
                                    'name': uploaded_file.name,
                                    'size': uploaded_file.size,
                                    'chunks': len(chunks)
                                }
                                st.session_state.uploaded_documents.append(doc_metadata)
                                save_document_metadata(st.session_state.uploaded_documents)
                                
                                st.success(f"Successfully processed {uploaded_file.name} ({len(chunks)} chunks)")
                            else:
                                st.error(f"Failed to extract text from {uploaded_file.name}")
                            
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Display uploaded documents
        if st.session_state.uploaded_documents:
            st.subheader("Uploaded Documents")
            for i, doc in enumerate(st.session_state.uploaded_documents):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['name']}")
                    st.caption(f"{doc['chunks']} chunks • {doc['size']:,} bytes")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        # Remove from vector store and metadata
                        st.session_state.vector_store.remove_document(doc['name'])
                        st.session_state.uploaded_documents.pop(i)
                        save_document_metadata(st.session_state.uploaded_documents)
                        st.rerun()
        else:
            st.info("No documents uploaded yet")
    
    # Main query interface
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.uploaded_documents:
        st.info("Please upload some aviation PDF documents to get started.")
        return
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the minimum weather requirements for VFR flight?",
        help="Ask questions about the content in your uploaded aviation documents"
    )
    
    if query:
        with st.spinner("Searching documents and generating response..."):
            try:
                response = st.session_state.query_engine.query(query)
                
                if response:
                    st.subheader("Response")
                    st.write(response['answer'])
                    
                    if response['sources']:
                        st.subheader("Sources")
                        for source in response['sources']:
                            with st.expander(f"📄 {source['document']} (Page {source['page']})"):
                                st.write(source['content'])
                                st.caption(f"Relevance Score: {source['score']:.3f}")
                else:
                    st.warning("No relevant information found in the uploaded documents.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Example queries
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What are the visibility requirements for different flight categories?
        - What documentation is required for flight planning?
        - What are the altitude restrictions in controlled airspace?
        - What weather minimums apply to instrument approaches?
        - What are the fuel reserve requirements for cross-country flights?
        """)

if __name__ == "__main__":
    main()
