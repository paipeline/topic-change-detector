import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go


def analyze_text(text, major_threshold, minor_threshold):
    """Call API for text analysis"""
    try:
        response = requests.post(
            "http://localhost:8005/analyze_text",
            json={
                "text": text,
                "major_threshold": major_threshold,
                "minor_threshold": minor_threshold,
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {str(e)}")
        return None

def plot_similarity_graph(changes):
    """Plot similarity changes graph"""
    if not changes:
        return None
    
    df = pd.DataFrame(changes)
    
    fig = go.Figure()
    
    # Add major change points
    major_changes = df[df['type'] == 'major']
    if not major_changes.empty:
        fig.add_trace(go.Scatter(
            x=major_changes['position'],
            y=major_changes['similarity'],
            mode='markers',
            name='Major Changes',
            marker=dict(size=12, color='#FF6B6B')
        ))
    
    # Add minor change points
    minor_changes = df[df['type'] == 'minor']
    if not minor_changes.empty:
        fig.add_trace(go.Scatter(
            x=minor_changes['position'],
            y=minor_changes['similarity'],
            mode='markers',
            name='Minor Changes',
            marker=dict(size=8, color='#FFD93D')
        ))
    
    fig.update_layout(
        title='Topic Changes Similarity Graph',
        xaxis_title='Sentence Position',
        yaxis_title='Similarity Score',
        yaxis_range=[0, 1],
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F0F0F0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F0F0F0')
    
    return fig

def get_sample_text():
    return """Deep learning has revolutionized artificial intelligence in recent years.
Neural networks are becoming increasingly sophisticated and powerful in their capabilities.
These models can now handle complex tasks with remarkable accuracy and efficiency.
Particularly in computer vision, CNNs have shown exceptional performance in image recognition.
While CNNs focus on spatial patterns, RNNs excel at sequential data processing.
The architecture of RNNs makes them particularly suitable for natural language tasks.

Moving to a completely different topic, let me share my weekend adventure at the zoo.
The pandas were absolutely adorable, playing with bamboo and rolling around.
The zookeeper gave an interesting presentation about panda conservation efforts.
We learned about their dietary habits and daily routines in captivity.

Returning to AI technology, let's discuss recent developments in language models.
The transformer architecture has largely replaced traditional RNNs in NLP tasks.
GPT and BERT represent significant milestones in language model development.
These models can understand context and generate human-like text with impressive accuracy.
However, their training requires massive computational resources and datasets.
The energy consumption of these large models has become an environmental concern.

Suddenly thinking about my recent hiking trip in the mountains.
The trail was challenging but rewarding, with breathtaking views at every turn.
We encountered several other hikers who shared their trail recommendations.
The sunset from the summit was an unforgettable experience.

Coming back to AI ethics and safety considerations.
The deployment of AI systems raises important questions about privacy and security.
We must carefully consider the societal implications of widespread AI adoption.
Regulatory frameworks need to evolve alongside technological advancements.
Ensuring AI system transparency has become increasingly crucial.
The challenge lies in balancing innovation with responsible development practices."""

def create_change_stub(text, max_length=100):
    """Create text preview"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def highlight_text_changes(text, changes):
    """Highlight change points in original text"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Create HTML formatted text, using different colors to mark change points
    html_parts = []
    for i, sentence in enumerate(sentences):
        # Check if this sentence is a change point
        is_major = any(c['position'] == i and c['type'] == 'major' for c in changes)
        is_minor = any(c['position'] == i and c['type'] == 'minor' for c in changes)
        
        if is_major:
            html_parts.append(f'<span style="background-color: #FF6B6B44; padding: 2px 4px; border-radius: 3px;">{sentence}</span>')
        elif is_minor:
            html_parts.append(f'<span style="background-color: #FFD93D44; padding: 2px 4px; border-radius: 3px;">{sentence}</span>')
        else:
            html_parts.append(sentence)
    
    return ".\n".join(html_parts)

def main():
    st.title("Topic Change Detector 🔍")
    
    # Sidebar parameters
    st.sidebar.header("Parameters")
    
    # Add sample text loading button to sidebar
    if st.sidebar.button("Load Sample Text"):
        st.session_state.text_input = get_sample_text()
    
    major_threshold = st.sidebar.slider(
        "Major Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.69,
        step=0.01
    )
    
    minor_threshold = st.sidebar.slider(
        "Minor Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.78,
        step=0.01
    )
    

    # Text input area
    text_input = st.text_area(
        "Input Text",
        height=300,
        key="text_input"
    )
    
    # Analyze button
    if st.button("Analyze Text ✨"):
        if not text_input:
            st.warning("⚠️ Please enter some text to analyze.")
            return
            
        with st.spinner("🔄 Analyzing text..."):
            results = analyze_text(
                text_input,
                major_threshold,
                minor_threshold,

            )
            
        if results:
            # 1. Display similarity graph
            st.header("📈 Similarity Graph")
            fig = plot_similarity_graph(results['changes'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # 2. Text visualization part
            st.header("📝 Text Visualization")
            st.markdown("""
            <style>
            .text-viz {
                line-height: 2;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .legend {
                display: flex;
                gap: 20px;
                margin-bottom: 10px;
                padding: 10px;
                background-color: white;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            .legend-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .color-box {
                width: 20px;
                height: 20px;
                border-radius: 3px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Add legend
            st.markdown("""
            <div class="legend">
                <div class="legend-item">
                    <div class="color-box" style="background-color: #FF6B6B44;"></div>
                    <span>Major Changes</span>
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: #FFD93D44;"></div>
                    <span>Minor Changes</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display highlighted text
            highlighted_text = highlight_text_changes(text_input, results['changes'])
            st.markdown(f'<div class="text-viz">{highlighted_text}</div>', unsafe_allow_html=True)
            
            # 3. Display statistical information card
            st.header("📊 Analysis Overview")
            cols = st.columns(3)
            with cols[0]:
                st.info(f"📑 Total Blocks: {results['statistics']['total_blocks']}")
            with cols[1]:
                st.error(f"🔴 Major Changes: {results['statistics']['major_changes']}")
            with cols[2]:
                st.warning(f"🟡 Minor Changes: {results['statistics']['minor_changes']}")
            
            # 4. Topic transition overview
            st.header("🔄 Topic Transitions")
            
            changes = results.get('changes', [])
            major_changes = [c for c in changes if c.get('type') == 'major']
            minor_changes = [c for c in changes if c.get('type') == 'minor']
            
            # Display major changes
            st.subheader(f"🔴 Major Changes ({len(major_changes)})")
            if major_changes:
                for change in major_changes:
                    with st.expander(f"Position {change['position']} (Similarity: {change['similarity']:.4f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Before:**")
                            st.markdown(f"""
                            <div style="padding: 10px; border-left: 3px solid #FF6B6B; background-color: #FF6B6B22;">
                                {create_change_stub(change.get('text_before', ''))}
                            </div>
                            <div style="margin-top: 5px;">
                                <b>Topics:</b> {', '.join(change.get('topic_before', {}).get('keywords', []))}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**After:**")
                            st.markdown(f"""
                            <div style="padding: 10px; border-left: 3px solid #FF6B6B; background-color: #FF6B6B22;">
                                {create_change_stub(change.get('text_after', ''))}
                            </div>
                            <div style="margin-top: 5px;">
                                <b>Topics:</b> {', '.join(change.get('topic_after', {}).get('keywords', []))}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No major changes detected in the text")
            
            # Display minor changes
            st.subheader(f"🟡 Minor Changes ({len(minor_changes)})")
            if minor_changes:
                for change in minor_changes:
                    with st.expander(f"Position {change['position']} (Similarity: {change['similarity']:.4f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Before:**")
                            st.markdown(f"""
                            <div style="padding: 10px; border-left: 3px solid #FFD93D; background-color: #FFD93D22;">
                                {create_change_stub(change.get('text_before', ''))}
                            </div>
                            <div style="margin-top: 5px;">
                                <b>Topics:</b> {', '.join(change.get('topic_before', {}).get('keywords', []))}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**After:**")
                            st.markdown(f"""
                            <div style="padding: 10px; border-left: 3px solid #FFD93D; background-color: #FFD93D22;">
                                {create_change_stub(change.get('text_after', ''))}
                            </div>
                            <div style="margin-top: 5px;">
                                <b>Topics:</b> {', '.join(change.get('topic_after', {}).get('keywords', []))}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No minor changes detected in the text")

if __name__ == "__main__":
    main() 