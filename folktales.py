import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer
from transformers import TFGPT2Model, TFGPT2LMHeadModel
import pickle
# Import the multiprocessing module



with open("merged_clean.txt", "r") as f:
  stories = f.read().split("\n\n\n\n")

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# # Create a CountVectorizer object and fit it on the original stories
# tfidf = TfidfVectorizer()
# tfidf.fit(stories)
# stories_weights = tfidf.transform(stories)


#stories_tokens = tokenizer(stories, padding=True, truncation=True, return_tensors="tf")

#model = TFGPT2Model.from_pretrained("gpt2")

# batch_size = 3  # adjust as needed
# n_samples = stories_tokens["input_ids"].shape[0]

# output_tensors = []
# for i in range(0, n_samples, batch_size):
#     input_batch = {
#         "input_ids": stories_tokens["input_ids"][i:i+batch_size],
#         "attention_mask": stories_tokens["attention_mask"][i:i+batch_size],
#     }
#     output_batch = model(input_batch)[0]
#     output_tensors.append(output_batch)
#     print(f"Processed {i+batch_size}/{n_samples} ({100*(i+batch_size)/n_samples:.2f}%) stories.")

# stories_encodings = tf.concat(output_tensors, axis=0)

# # Saving
# with open("stories_encodings.pkl", "wb") as f:
#     pickle.dump(stories_encodings, f)

# #reload encodings

# # Loading
with open("stories_encodings.pkl", "rb") as f:
    stories_encodings = pickle.load(f)

# stories_encodings_np = stories_encodings.numpy()
# stories_encodings_2d = np.reshape(stories_encodings_np, (stories_encodings_np.shape[0], -1))


# print("loaded ")

# # Apply K-means clustering to the encodings; a form of topic modeling
num_clusters = 10  # adjust as needed
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# kmeans.fit(stories_encodings_2d)

# with open("stories_encodings.pkl", "wb") as f:
#     pickle.dump(kmeans, f)

model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Get the labels assigned by K-means to each encoding
labels = kmeans.labels_

# Print the number of stories assigned to each topic
print('Number of stories per topic:')
print(pd.Series(labels).value_counts())

# Select a cluster of interest
cluster_id = 0

# Get the anchor encoding for the selected cluster
anchor_encodings = kmeans.cluster_centers_

# Get the anchor encoding for the selected cluster
anchor_encodings = kmeans.cluster_centers_

# Generate new text using the anchor encoding as a prompt
for i, anchor_encoding in enumerate(anchor_encodings):
    # Decode the anchor encoding into text
    anchor_text = tokenizer.decode(anchor_encoding)
    if anchor_encoding is not None:
        anchor_text = tokenizer.decode(anchor_encoding)
    else:
        anchor_text = ""
    print(f"Anchor {i}: {anchor_text}")
    
    # Generate new text using the anchor encoding as a prompt
    generated_text = model.generate(
        input_ids=tokenizer.encode(anchor_text, return_tensors='tf'),
        max_length=100,
        num_return_sequences=1,
        temperature=1.0
    )

    # Decode the generated text
    generated_text = tokenizer.decode(generated_text[0])
    print(f"Generated text {i}: {generated_text}")
