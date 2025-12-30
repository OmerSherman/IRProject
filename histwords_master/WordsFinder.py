from representations.sequentialembedding import SequentialEmbedding
if __name__ == "__main__":

    fiction_embeddings = SequentialEmbedding.load("embeddings/sgns", list(range(1810, 2000, 10)))

    words  =fiction_embeddings.get_seq_closest("gay",1810, 110 )
    print(words)