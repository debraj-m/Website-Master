import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import time
from urllib.parse import urljoin, urlparse
warnings.filterwarnings('ignore')

class StreamlitWebsiteRAG:
    def __init__(self):
        self.documents = []
        self.sentences = []
        self.raw_content = ""
        # Optimized TF-IDF configuration
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Better tokenization
        )
        self.sentence_embeddings = None
        
        # Enhanced query expansion
        self.query_expansions = {
            'services': ['service', 'services', 'offer', 'offers', 'provide', 'provides', 'solution', 'solutions', 'product', 'products', 'business', 'work', 'specialize', 'specializes', 'expertise', 'capabilities'],
            'about': ['about', 'company', 'organization', 'business', 'who', 'what', 'mission', 'vision', 'description', 'overview', 'introduction', 'history', 'story'],
            'contact': ['contact', 'contacts', 'email', 'phone', 'address', 'reach', 'call', 'message', 'location', 'office', 'headquarters'],
            'price': ['price', 'prices', 'cost', 'costs', 'pricing', 'fee', 'fees', 'rate', 'rates', 'charge', 'charges', 'payment', 'payments'],
            'team': ['team', 'staff', 'employees', 'people', 'members', 'leadership', 'management', 'founders', 'ceo', 'director'],
            'technology': ['technology', 'tech', 'software', 'platform', 'system', 'tool', 'tools', 'framework', 'development'],
            'industry': ['industry', 'sector', 'market', 'field', 'domain', 'vertical', 'niche']
        }
        
    def scrape_website(self, url):
        """Enhanced website scraping with better content extraction"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                               'noscript', 'iframe', 'form', 'button', 'input']):
                element.decompose()
            
            # Extract structured content
            content_sections = []
            
            # 1. Page title
            title = soup.find('title')
            if title:
                content_sections.append(f"PAGE_TITLE: {title.get_text().strip()}")
            
            # 2. Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_sections.append(f"META_DESCRIPTION: {meta_desc['content'].strip()}")
            
            # 3. Headers (h1, h2, h3) - these are very important
            for i, header in enumerate(soup.find_all(['h1', 'h2', 'h3'], limit=20)):
                text = header.get_text().strip()
                if text and len(text) > 3:
                    content_sections.append(f"HEADER_{i+1}: {text}")
            
            # 4. Main content areas
            main_content = []
            
            # Look for main content containers
            main_selectors = [
                'main', 'article', '[role="main"]', '.main-content', '#main-content',
                '.content', '#content', '.post-content', '.entry-content'
            ]
            
            main_container = None
            for selector in main_selectors:
                main_container = soup.select_one(selector)
                if main_container:
                    break
            
            if not main_container:
                main_container = soup.find('body')
            
            # Extract paragraphs and list items
            if main_container:
                for element in main_container.find_all(['p', 'li', 'div', 'span'], limit=100):
                    text = element.get_text().strip()
                    # Filter out very short or very long text
                    if 10 <= len(text) <= 1000 and len(text.split()) >= 3:
                        # Check if it's not just navigation or boilerplate
                        if not self._is_boilerplate(text):
                            main_content.append(text)
            
            # 5. Combine all content
            all_content = content_sections + main_content
            
            # Clean and process text
            processed_content = []
            for text in all_content:
                # Clean text
                text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                text = re.sub(r'[^\w\s.,!?;:()\-\'"\/]', '', text)  # Remove special chars
                text = text.strip()
                
                if text and len(text) > 5:
                    processed_content.append(text)
            
            final_content = '\n'.join(processed_content)
            self.raw_content = final_content
            
            return final_content
            
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {str(e)}"
        except Exception as e:
            return None, f"Parsing error: {str(e)}"
    
    def _is_boilerplate(self, text):
        """Check if text is likely boilerplate/navigation"""
        boilerplate_indicators = [
            'cookie', 'privacy policy', 'terms of service', 'all rights reserved',
            'copyright', 'menu', 'navigation', 'skip to', 'click here',
            'read more', 'learn more', 'sign up', 'subscribe', 'follow us'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in boilerplate_indicators)
    
    def expand_query(self, query):
        """Enhanced query expansion"""
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add related terms
        for key, expansions in self.query_expansions.items():
            if key in query_lower or any(term in query_lower for term in expansions[:3]):
                expanded_terms.extend(expansions[:5])  # Limit expansion
        
        # Add keyword variations
        words = query_lower.split()
        for word in words:
            if len(word) > 4:  # Only expand longer words
                # Add plural/singular forms
                if word.endswith('s') and len(word) > 5:
                    expanded_terms.append(word[:-1])  # Remove 's'
                else:
                    expanded_terms.append(word + 's')  # Add 's'
        
        return ' '.join(set(expanded_terms))  # Remove duplicates
    
    def advanced_sentence_split(self, text):
        """Advanced sentence splitting with better handling"""
        # First, split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Further split by line breaks and colons in structured content
        refined_sentences = []
        for sentence in sentences:
            # Split by line breaks for structured content
            parts = sentence.split('\n')
            for part in parts:
                part = part.strip()
                if part:
                    # Split by colons for definition-like content
                    if ':' in part and len(part.split(':')) == 2:
                        refined_sentences.extend([p.strip() for p in part.split(':') if p.strip()])
                    else:
                        refined_sentences.append(part)
        
        # Filter and clean sentences
        cleaned_sentences = []
        for sentence in refined_sentences:
            sentence = sentence.strip()
            # More lenient filtering
            if (5 <= len(sentence) <= 800 and 
                len(sentence.split()) >= 2 and 
                not sentence.startswith(('http', 'www', '@', '#'))):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def intelligent_chunking(self, text, chunk_size=300, overlap=50):
        """Intelligent text chunking with context preservation"""
        sentences = self.advanced_sentence_split(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_size = min(overlap, len(current_chunk))
                if overlap_size > 0:
                    overlap_sentences = current_chunk[-overlap_size:]
                    current_chunk = overlap_sentences + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def process_url(self, url):
        """Process URL with enhanced error handling"""
        try:
            # Scrape website
            result = self.scrape_website(url)
            if isinstance(result, tuple):
                return False, result[1]
            
            content = result
            if not content or len(content.strip()) < 50:
                return False, "Insufficient content found on the website"
            
            # Create chunks
            chunks = self.intelligent_chunking(content)
            self.documents = chunks
            
            # Create sentence-level data for retrieval
            all_sentences = []
            for chunk in chunks:
                chunk_sentences = self.advanced_sentence_split(chunk)
                all_sentences.extend(chunk_sentences)
            
            # Filter sentences more carefully
            self.sentences = []
            for sentence in all_sentences:
                # More inclusive filtering
                if (5 <= len(sentence) <= 500 and 
                    len(sentence.split()) >= 2 and
                    not self._is_boilerplate(sentence)):
                    self.sentences.append(sentence)
            
            if len(self.sentences) == 0:
                return False, "No valid sentences found after processing"
            
            # Create embeddings
            try:
                # Combine sentences and chunks for embedding
                all_text = self.sentences + chunks
                self.sentence_embeddings = self.vectorizer.fit_transform(all_text)
                
                return True, f"Successfully processed {len(self.sentences)} sentences and {len(chunks)} chunks"
                
            except Exception as e:
                return False, f"Error creating embeddings: {str(e)}"
                
        except Exception as e:
            return False, f"Processing error: {str(e)}"
    
    def retrieve_relevant_content(self, query, top_k=10):
        """Enhanced content retrieval"""
        if self.sentence_embeddings is None:
            return []
        
        try:
            # Expand query
            expanded_query = self.expand_query(query)
            
            # Create embeddings for both queries
            original_embedding = self.vectorizer.transform([query])
            expanded_embedding = self.vectorizer.transform([expanded_query])
            
            # Calculate similarities
            similarities_orig = cosine_similarity(original_embedding, self.sentence_embeddings).flatten()
            similarities_exp = cosine_similarity(expanded_embedding, self.sentence_embeddings).flatten()
            
            # Combine similarities with weights
            similarities = 0.6 * similarities_orig + 0.4 * similarities_exp
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
            
            # Filter by minimum similarity threshold
            relevant_content = []
            min_similarity = 0.05  # Lower threshold for better recall
            
            all_content = self.sentences + self.documents
            
            for idx in top_indices:
                if similarities[idx] > min_similarity:
                    content_type = "sentence" if idx < len(self.sentences) else "chunk"
                    relevant_content.append({
                        'text': all_content[idx],
                        'score': similarities[idx],
                        'type': content_type
                    })
            
            # Sort by score and return top k
            relevant_content = sorted(relevant_content, key=lambda x: x['score'], reverse=True)
            return relevant_content[:top_k]
            
        except Exception as e:
            st.error(f"Error retrieving content: {str(e)}")
            return []
    
    def answer_question(self, question):
        """Generate answer with improved logic"""
        if not self.sentences and not self.documents:
            return "No website content has been processed yet. Please provide a URL first.", 0
        
        # Retrieve relevant content
        relevant_content = self.retrieve_relevant_content(question, top_k=8)
        
        if not relevant_content:
            # Fallback: try with a more general search
            general_terms = ['service', 'company', 'business', 'about', 'what', 'who', 'how']
            for term in general_terms:
                if term in question.lower():
                    relevant_content = self.retrieve_relevant_content(term, top_k=5)
                    if relevant_content:
                        break
        
        if not relevant_content:
            return "I couldn't find relevant information to answer your question. The website might not contain information about this topic, or try rephrasing your question.", 0
        
        # Generate answer
        answer, confidence = self.generate_comprehensive_answer(question, relevant_content)
        return answer, confidence
    
    def generate_comprehensive_answer(self, question, relevant_content):
        """Generate a comprehensive answer"""
        if not relevant_content:
            return "No relevant information found.", 0
        
        # Separate high and medium confidence content
        high_conf = [c for c in relevant_content if c['score'] > 0.15]
        medium_conf = [c for c in relevant_content if 0.08 <= c['score'] <= 0.15]
        
        if not high_conf and not medium_conf:
            high_conf = relevant_content[:3]  # Take top 3 if none meet threshold
        
        # Build answer
        answer_parts = []
        
        # Primary answer from best content
        primary_content = high_conf[0] if high_conf else relevant_content[0]
        answer_parts.append(primary_content['text'])
        
        # Add supporting information
        supporting_content = []
        used_content = {primary_content['text']}
        
        for content in (high_conf[1:] + medium_conf)[:4]:
            if content['text'] not in used_content:
                # Check for significant overlap
                if self.calculate_text_similarity(primary_content['text'], content['text']) < 0.7:
                    supporting_content.append(content['text'])
                    used_content.add(content['text'])
        
        if supporting_content:
            answer_parts.append(f"\n\nAdditional relevant information:")
            for i, content in enumerate(supporting_content, 1):
                answer_parts.append(f"{i}. {content}")
        
        # Calculate confidence
        avg_confidence = np.mean([c['score'] for c in relevant_content[:3]])
        
        return '\n'.join(answer_parts), avg_confidence
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        try:
            embeddings = self.vectorizer.transform([text1, text2])
            similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
            return similarity
        except:
            # Fallback to word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total if total > 0 else 0
    
    def get_summary(self):
        """Enhanced content summary"""
        if not self.sentences and not self.documents:
            return "No content processed yet."
        
        summary_parts = []
        
        # Extract key information
        key_sentences = []
        for sentence in self.sentences[:20]:  # Check first 20 sentences
            if any(marker in sentence for marker in ['PAGE_TITLE:', 'META_DESCRIPTION:', 'HEADER_']):
                key_sentences.append(sentence)
        
        if key_sentences:
            summary_parts.append("**Key Information:**")
            for sentence in key_sentences[:5]:
                clean_sentence = re.sub(r'^(PAGE_TITLE:|META_DESCRIPTION:|HEADER_\d+:)\s*', '', sentence)
                summary_parts.append(f"• {clean_sentence}")
        
        # Add general content overview
        if self.sentences:
            summary_parts.append(f"\n**Content Overview:**")
            content_sentences = [s for s in self.sentences if not any(marker in s for marker in ['PAGE_TITLE:', 'META_DESCRIPTION:', 'HEADER_'])]
            for sentence in content_sentences[:3]:
                summary_parts.append(f"• {sentence}")
        
        stats = f"\n**Processing Stats:**\n• {len(self.sentences)} sentences processed\n• {len(self.documents)} content chunks created"
        summary_parts.append(stats)
        
        return '\n'.join(summary_parts)

# Rest of the Streamlit interface remains the same
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = StreamlitWebsiteRAG()
    if 'processed_url' not in st.session_state:
        st.session_state.processed_url = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    st.set_page_config(
        page_title="Enhanced Website Q&A RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.title("🔍 Enhanced Website Q&A RAG System")
    st.markdown("Ask questions about any website content using advanced Retrieval-Augmented Generation")
    
    with st.sidebar:
        st.header("📝 Website Input")
        
        url_input = st.text_input(
            "Enter Website URL:",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to analyze"
        )
        
        if st.button("🚀 Process Website", disabled=st.session_state.processing):
            if url_input.strip():
                if not url_input.startswith(('http://', 'https://')):
                    url_input = 'https://' + url_input
                
                st.session_state.processing = True
                
                with st.spinner("Processing website... This may take a moment."):
                    success, message = st.session_state.rag_system.process_url(url_input)
                
                st.session_state.processing = False
                
                if success:
                    st.session_state.processed_url = url_input
                    st.success("✅ Website processed successfully!")
                    st.info(message)
                    st.session_state.chat_history = []
                else:
                    st.error(f"❌ Failed to process website: {message}")
            else:
                st.warning("Please enter a valid URL")
        
        if st.session_state.processed_url:
            st.success(f"📄 **Current Website:**\n{st.session_state.processed_url}")
            
            if st.button("📋 Show Content Summary"):
                summary = st.session_state.rag_system.get_summary()
                st.info(summary)
        
        if st.button("🗑️ Clear All"):
            st.session_state.rag_system = StreamlitWebsiteRAG()
            st.session_state.processed_url = None
            st.session_state.chat_history = []
            st.success("Cleared all data!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What services does this company offer?",
            disabled=not st.session_state.processed_url
        )
        
        ask_col1, ask_col2 = st.columns([1, 4])
        with ask_col1:
            ask_button = st.button("🤔 Ask", disabled=not st.session_state.processed_url or not question.strip())
        
        if ask_button and question.strip():
            with st.spinner("Finding answer..."):
                answer, confidence = st.session_state.rag_system.answer_question(question)
            
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'timestamp': time.strftime("%H:%M:%S")
            })
        
        if st.session_state.chat_history:
            st.header("📋 Q&A History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:60]}{'...' if len(chat['question']) > 60 else ''}", expanded=(i==0)):
                    
                    st.markdown(f"**❓ Question:** {chat['question']}")
                    
                    confidence_color = "🟢" if chat['confidence'] > 0.15 else "🟡" if chat['confidence'] > 0.08 else "🔴"
                    confidence_text = "High" if chat['confidence'] > 0.15 else "Medium" if chat['confidence'] > 0.08 else "Low"
                    
                    st.markdown(f"**✅ Answer:** {chat['answer']}")
                    st.markdown(f"**📊 Confidence:** {confidence_color} {confidence_text} ({chat['confidence']:.3f})")
                    st.markdown(f"**🕒 Time:** {chat['timestamp']}")
    
    with col2:
        st.header("ℹ️ Instructions")
        
        instructions = """
        ### How to use:
        
        1. **Enter URL**: Put any website URL in the sidebar
        2. **Process**: Click "Process Website" to analyze the content
        3. **Ask Questions**: Type questions about the website content
        4. **Get Answers**: The system will find relevant information and provide answers
        
        ### Enhanced Features:
        - **Better Content Extraction**: Prioritizes headers, titles, and main content
        - **Smart Text Processing**: Filters out navigation and boilerplate text
        - **Advanced Query Expansion**: Automatically adds related terms
        - **Improved Chunking**: Preserves context while splitting content
        - **Lower Similarity Thresholds**: Better recall for finding relevant content
        
        ### Tips for Better Results:
        - Use specific questions: "What services does this company offer?"
        - Try different phrasings if you don't get good results
        - Ask about specific topics: "Who is the team?" or "What technology do they use?"
        - Check the confidence score - the system is more lenient now
        
        ### Example Questions:
        - "What does this company do?"
        - "What services are offered?"
        - "Who are the founders or team members?"
        - "What industries do they serve?"
        - "How can I contact them?"
        - "What technology do they use?"
        """
        
        st.markdown(instructions)
        
        if st.session_state.processed_url:
            st.header("📊 Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
            st.metric("Sentences Processed", len(st.session_state.rag_system.sentences))
            st.metric("Document Chunks", len(st.session_state.rag_system.documents))

if __name__ == "__main__":
    main()