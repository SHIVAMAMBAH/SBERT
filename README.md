## SBERT
- **Model Overview**: SBERT is a modification of the BERT (Bidirectional Encoder Representations from Transformers) model, which is a transformer-based architecture. The main difference is that BERT is specifically optimized for sentence-level tasks rather than token-level tasks.
- **Encoding Sentences**: SBERT works by taking an input sentence and transforming them into fixed-size vector embeddings.
- **Fixed-Length Sentence Embeddings**: The output vector is a fixed-length numeric vector (usually 512 or 768 dimensions depending on the model). This vector is a high-dimensional representations of the input sentence and captures both the syntax ans sementics of the text.

## Cosine Similarity
Once we have the vectors representation of the sentences, we can use cosine similarity to measure the "similarity" between two sentences.

![Capture-13](https://github.com/user-attachments/assets/2cdec760-20a5-468f-b78e-e12d02326ade)

The cosine similarity will always be a value between -1 and 1.
- **1** means the vectors are identical. (the sentences are very similar)
- **0** means the vectors are orthogonal. (the sentences are completely dissimilar)
- **-1** means the vectors are completely opposite. (the sentences are very dissimilar, though in the case of sentences embeddings, value closer to -1 are rare).
