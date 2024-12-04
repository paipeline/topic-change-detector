from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from topic_change_detector import TopicChangeDetector
import logging

# 在文件开头配置日志
logging.basicConfig(level=logging.INFO)

app = FastAPI()
detector = TopicChangeDetector()

class TextAnalysisRequest(BaseModel):
    text: str
    major_threshold: Optional[float] = 0.77
    minor_threshold: Optional[float] = 0.83
    use_stopwords: Optional[bool] = True

class TextBlockInfo(BaseModel):
    content: str
    start_line: int
    end_line: int
    change_type: Optional[str] = None
    similarity: Optional[float] = None

class TextAnalysisResponse(BaseModel):
    blocks: List[TextBlockInfo]
    changes: List[dict]
    statistics: dict

@app.post("/analyze_text", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    try:
        logging.info("Received request: %s", request.json())
        
        # 1. Preprocess text: split by periods
        sentences = [s.strip() for s in request.text.split('.') if s.strip()]
        logging.info("Preprocessed sentences: %s", sentences)
        
        # 2. Detect topic changes
        changes = detector.detect_topic_changes(
            text=request.text,
            major_threshold=request.major_threshold,
            minor_threshold=request.minor_threshold,

        )
        logging.info("Detected changes: %s",
        # Convert numpy data types to native Python types  changes)
        and add adjacent sentences
        converted_changes = []
        for change in changes:
            pos = change['position']
            converted_change = {
                'position': int(change['position']),
                'type': str(change['type']),
                'similarity': float(change['similarity']),
                'text_before': sentences[pos-1] if pos > 0 else "",  # Add previous sentence
                'text_after': sentences[pos] if pos < len(sentences) else "",  # Add current sentence
                'topic_before': change.get('topic_before', {'keywords': []}),
                'topic_after': change.get('topic_after', {'keywords': []})
            }
            converted_changes.append(converted_change)
        
        logging.info("Converted changes: %s", converted_changes)
        
        # 3. Divide text into blocks based on topic change points
        blocks = []
        current_block_start = 0
        
        # Process according to detected change points order
        change_positions = sorted([c['position'] for c in converted_changes])
        
        for i, pos in enumerate(change_positions):
            # Find information of the current change point
            change_info = next(c for c in converted_changes if c['position'] == pos)
            
            # Add current block
            block_content = '. '.join(sentences[current_block_start:pos]) + '.'
            blocks.append(TextBlockInfo(
                content=block_content,
                start_line=current_block_start,
                end_line=pos-1,
                change_type=change_info['type'],
                similarity=change_info['similarity']
            ))
            current_block_start = pos
        
        # Add the last block
        if current_block_start < len(sentences):
            block_content = '. '.join(sentences[current_block_start:]) + '.'
            blocks.append(TextBlockInfo(
                content=block_content,
                start_line=current_block_start,
                end_line=len(sentences)-1,
                change_type=None,  # The last block has no change type
                similarity=None
            ))
        
        # 4. Generate statistics
        statistics = {
            'total_blocks': len(blocks),
            'major_changes': len([c for c in converted_changes if c['type'] == 'major']),
            'minor_changes': len([c for c in converted_changes if c['type'] == 'minor'])
        }
        
        logging.info("Generated statistics: %s", statistics)
        
        return TextAnalysisResponse(
            blocks=blocks,
            changes=converted_changes,
            statistics=statistics
        )
        
    except Exception as e:
        logging.error("Error processing request: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for testing
@app.get("/test_analysis")
async def test_analysis():
    test_text = """
    Apache Kafka is a distributed streaming platform with high scalability and fault tolerance.
    The platform uses a publish-subscribe model and supports real-time data stream processing.
    
    Yesterday I went to the zoo and saw pandas, they were so adorable.
    The red panda was climbing up and down the bamboo, looking very cute.
    
    Back to Kafka's message reliability, it provides various mechanisms to ensure data safety.
    Producers can send messages synchronously or asynchronously.
    """
    
    request = TextAnalysisRequest(text=test_text)
    return await analyze_text(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)