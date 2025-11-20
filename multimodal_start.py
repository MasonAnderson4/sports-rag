import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")


# create a chromadb object
chroma_client = chromadb.PersistentClient(path="./data/chroma.db")

# instantiate image loader
image_loader = ImageLoader()

# instantiate multimodal embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# create the collection, - vector database
collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)


# check the count of the collection
# print(collection.count())  # res: 2

# Use .add() to add a new record or .update() to update existing record
# on first run add() is used, on subsequent runs update() is used
collection.add(
    ids=[
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
    ],
    uris=[
        
        "./images/archery.jpg",
        "./images/baseball.jpg",
        "./images/basketball.jpg",
        "./images/bowling.jpg",
        "./images/discgolf.jpg",
        "./images/f1.jpg",
        "./images/fieldhockey.jpg",
        "./images/hockey.jpg",
        "./images/jousting.jpg",
        "./images/snowboarding.jpg",
    ],
    metadatas=[
        {
            "item_id": "1",
            "category": "sports",
            "item_name": "Archery image",
        },
        {
            "item_id": "2",
            "category": "sport",
            "item_name": "Baseball image",
        },
        {
            "item_id": "3",
            "category": "sport",
            "item_name": "Basketball image",
        },
        {
            "item_id": "4",
            "category": "sport",
            "item_name": "bowling image",
        },
        {
            "item_id": "5",
            "category": "sport",
            "item_name": "discgolf image",
        },
        {
            "item_id": "6",
            "category": "sport",
            "item_name": "f1 image",
        },
        {
            "item_id": "7",
            "category": "sport",
            "item_name": "fieldhockey image",
        },
        {
            "item_id": "8",
            "category": "sport",
            "item_name": "hockey image",
        },
        {
            "item_id": "9",
            "category": "sport",
            "item_name": "jousting image",
        },
        {
            "item_id": "10",
            "category": "sport",
            "item_name": "snowboarding image",
        },
    ],
)


# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict) -> None:
    result_count = len(query_results["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}"
            )

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()


# It is possible to submit multiple queries at the same time, just add to the list.
query_texts = ["sports, f1"]

# Query vector db - return 2 results
query_results = collection.query(
    query_texts=query_texts,
    n_results=2,
    include=["documents", "distances", "metadatas", "data", "uris"],
    # where={"category": "animal"}, # filter by metadata - optional - first run remove this
)

print_query_results(query_texts, query_results)

