from pymilvus import model
from collection import client

# If connection to https://huggingface.co/ failed, uncomment the following path
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
#print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# Each entity has id, vector representation, raw text, and a subject label that we use
# to demo metadata filtering later.
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

#print("Data has", len(data), "entities, each with fields: ", data[0].keys())
#print("Vector dim:", len(data[0]["vector"]))
res = client.insert(collection_name="demo_collection", data=data)
#print(res)

docs2 = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
vectors2 = embedding_fn.encode_documents(docs)
data2 = [
    {"id": 3 + i, "vector": vectors2[i], "text": docs2[i], "subject": "biology"}
    for i in range(len(vectors))
]

client.insert(collection_name="demo_collection", data=data2)

# This will exclude any text in "history" subject despite close to the query vector.
res2 = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"],
)

#print(res2)

#print("print history topics..")
res3 = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)

#print(res3)
#query
print("executing query for history topics...")
res4 = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)

print(res4)

print("executing query2 for biology topics...")
res5 = client.query(
    collection_name="demo_collection",
    filter="subject == 'biology'",
    output_fields=["text", "subject"],
)

print(res5)

print("executing query3 for sports topics...")
res6 = client.query(
    collection_name="demo_collection",
    filter="subject == 'sports'",
    output_fields=["text", "subject"],
)

print(res6)
