# Sports-RAG 🏀⚽🏒

A multimodal RAG (Retrieval-Augmented Generation) system that enables semantic search over sports images using natural language queries. Built with ChromaDB and OpenCLIP for powerful image-to-text and text-to-image similarity matching.

## Overview

Sports-RAG demonstrates multimodal embedding and retrieval by indexing sports images and allowing users to search them using natural language descriptions. The system uses OpenCLIP's vision-language model to understand both images and text in a shared embedding space, enabling intuitive image discovery.

## Features

- **Multimodal Embeddings**: Uses OpenCLIP to create unified embeddings for both images and text
- **Semantic Image Search**: Find images using natural language descriptions
- **Persistent Vector Storage**: ChromaDB for efficient storage and retrieval
- **Visual Results Display**: Automatic image visualization with matplotlib
- **Metadata Filtering**: Query refinement using custom metadata fields
- **Batch Query Support**: Process multiple queries simultaneously

## Technology Stack

- **ChromaDB**: Vector database for embedding storage and similarity search
- **OpenCLIP**: Open-source vision-language model for multimodal embeddings
- **Matplotlib**: Image visualization and display
- **Python 3.8+**: Core programming language

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MasonAnderson4/sports-rag.git
cd sports-rag
```

2. Install required dependencies:
```bash
pip install chromadb matplotlib pillow open-clip-torch
```

3. Create the required directory structure:
```bash
mkdir -p images data
```

4. Add your sports images to the `./images/` directory with names matching those in the code:
   - archery.jpg
   - baseball.jpg
   - basketball.jpg
   - bowling.jpg
   - discgolf.jpg
   - f1.jpg
   - fieldhockey.jpg
   - hockey.jpg
   - jousting.jpg
   - snowboarding.jpg

## Usage

### Basic Usage

Run the script to index images and perform a search:

```bash
python multimodal_start.py
```

The default query searches for "sports, f1" and returns the top 2 most similar images.

### Custom Queries

Modify the `query_texts` list to search for different sports or activities:

```python
query_texts = ["winter sports"]  # Find winter sports like snowboarding, hockey
query_texts = ["ball sports"]    # Find sports involving balls
query_texts = ["racing, speed"]  # Find fast-paced sports like F1
```

### Adjusting Result Count

Change the `n_results` parameter to return more or fewer matches:

```python
query_results = collection.query(
    query_texts=query_texts,
    n_results=5,  # Return top 5 results instead of 2
    ...
)
```

### Metadata Filtering

Filter results by metadata categories:

```python
query_results = collection.query(
    query_texts=query_texts,
    n_results=2,
    where={"category": "sport"},  # Only return items in "sport" category
    ...
)
```

## How It Works

1. **Image Loading**: Sports images are loaded from the `./images/` directory
2. **Embedding Generation**: OpenCLIP creates vector embeddings for each image
3. **Storage**: Embeddings are stored in ChromaDB with associated metadata
4. **Query Processing**: User text queries are converted to embeddings
5. **Similarity Search**: ChromaDB finds the most similar image embeddings
6. **Results Display**: Matching images are displayed with their metadata and similarity scores

## Project Structure

```
sports-rag/
├── multimodal_start.py  # Main application script
|-- venv/
├── images/              # Directory for sports images
│   ├── archery.jpg
│   ├── baseball.jpg
│   ├── basketball.jpg
│   └── ...
├── data/
│   └── chroma.db/      # Persistent ChromaDB storage
└── README.md
```

## Example Output

```
Results for query: sports, f1
id: 6, distance: 0.234, metadata: {'item_id': '6', 'category': 'sport', 'item_name': 'f1 image'}, document: None
data: ./images/f1.jpg
[Image displayed]

id: 10, distance: 0.456, metadata: {'item_id': '10', 'category': 'sport', 'item_name': 'snowboarding image'}, document: None
data: ./images/snowboarding.jpg
[Image displayed]
```

## Understanding Distance Scores

- **Lower distance = Higher similarity**
- Distance of 0.0 = Perfect match
- Distance < 0.5 = Very similar
- Distance > 1.0 = Less similar

## Customization

### Adding New Images

1. Add new image files to the `./images/` directory
2. Update the `ids`, `uris`, and `metadatas` lists in `main.py`:

```python
collection.add(
    ids=["11"],
    uris=["./images/soccer.jpg"],
    metadatas=[{
        "item_id": "11",
        "category": "sport",
        "item_name": "soccer image",
    }],
)
```

### Changing the Embedding Model

Modify the embedding function to use a different OpenCLIP model:

```python
embedding_function = OpenCLIPEmbeddingFunction(
    model_name="ViT-B-32",
    checkpoint="laion2b_s34b_b79k"
)
```

### Custom Metadata Fields

Add additional metadata for more sophisticated filtering:

```python
metadatas=[{
    "item_id": "1",
    "category": "sport",
    "item_name": "Archery image",
    "difficulty": "medium",
    "location": "outdoor",
    "equipment_required": True
}]
```

## Use Cases

- **Sports Content Discovery**: Find similar sports images in large databases
- **Athletic Activity Classification**: Organize sports images by type
- **Training Data Curation**: Build datasets for sports recognition models
- **Content Recommendation**: Suggest related sports content to users
- **Visual Search Engine**: Create a sports image search platform

## Limitations

- Requires images to be stored locally
- Embedding quality depends on OpenCLIP model used
- Large image collections require significant storage space
- Query accuracy depends on how well images match natural language descriptions

## Future Enhancements

- [ ] Add support for video frame analysis
- [ ] Implement real-time web camera search
- [ ] Create web interface for image uploads and searches
- [ ] Add image preprocessing and augmentation
- [ ] Support for multiple languages in queries
- [ ] Implement relevance feedback for query refinement
- [ ] Add batch image upload functionality
- [ ] Create API endpoint for integration with other applications

## Troubleshooting

**Images not displaying**: Ensure image files exist at the specified paths and are valid image formats (JPG, PNG).

**ChromaDB errors**: Delete the `./data/chroma.db/` directory and re-run to recreate the database.

**Memory issues**: Reduce the number of images or use a smaller embedding model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI's CLIP model for vision-language understanding
- ChromaDB for efficient vector storage
- OpenCLIP for open-source CLIP implementation

