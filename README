# Evaluation Metrics for Retriever

## 1. Precision@K
What is Precision@K?
•	Precision@K measures the proportion of relevant documents among the top K documents retrieved by the retriever.
•	Formula: Precision@K=Number of Relevant Documents in Top KK\text{Precision@K} = \frac{\text{Number of Relevant Documents in Top K}}{K}Precision@K=KNumber of Relevant Documents in Top K
How it Works?
•	The retriever ranks documents based on their relevance scores.
•	Precision@K calculates the ratio of relevant documents in the top K results.
When to Choose This Metric?
•	Use Precision@K when precision is critical, such as in scenarios where false positives (irrelevant documents) can degrade system performance.
•	Example: In question-answering systems where irrelevant documents could mislead the generator.
________________________________________
## 2. Recall@K
What is Recall@K?
•	Recall@K measures the proportion of relevant documents retrieved out of all relevant documents available in the dataset.
•	Formula: Recall@K=Number of Relevant Documents in Top KTotal Number of Relevant Documents\text{Recall@K} = \frac{\text{Number of Relevant Documents in Top K}}{\text{Total Number of Relevant Documents}}Recall@K=Total Number of Relevant DocumentsNumber of Relevant Documents in Top K
How it Works?
•	It calculates how many relevant documents the retriever successfully retrieves within the top K.
•	A higher Recall@K means fewer relevant documents are missed.
When to Choose This Metric?
•	Use Recall@K when coverage of relevant documents is more important than precision.
•	Example: In applications like knowledge base search, where missing relevant documents is unacceptable.
________________________________________
## 3. Mean Reciprocal Rank (MRR)
What is MRR?
•	Mean Reciprocal Rank evaluates how quickly the first relevant document appears in the ranked list of results.
•	Formula: MRR=1N∑i=1N1Rank of the First Relevant Documenti\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{Rank of the First Relevant Document}_i}MRR=N1i=1∑NRank of the First Relevant Documenti1 where NNN is the number of queries.
How it Works?
•	For each query, MRR assigns a score based on the rank of the first relevant document. If a relevant document appears at rank 1, the score is 1; if at rank 3, the score is 13\frac{1}{3}31, and so on.
•	MRR is the average of these scores across all queries.
When to Choose This Metric?
•	Use MRR when early retrieval of relevant documents is crucial, such as in scenarios where users are only likely to interact with the top result or few results.
•	Example: FAQ retrieval systems, where users expect the first result to be highly relevant.
________________________________________
## 4. Normalized Discounted Cumulative Gain (NDCG)
What is NDCG?
•	NDCG evaluates the quality of ranked retrieval results by considering both the relevance and the position of the retrieved documents.
•	Formula: DCG@K=∑i=1KRelevance of Documentilog⁡2(i+1)\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{Relevance of Document}_i}{\log_2(i + 1)}DCG@K=i=1∑Klog2(i+1)Relevance of Documenti NDCG@K=DCG@KIdeal DCG@K\text{NDCG@K} = \frac{\text{DCG@K}}{\text{Ideal DCG@K}}NDCG@K=Ideal DCG@KDCG@K where Ideal DCG represents the best possible ranking of relevant documents.
How it Works?
•	Assigns a higher weight to relevant documents appearing earlier in the ranking.
•	Compares the actual ranking with the ideal ranking to calculate a normalized score.
When to Choose This Metric?
•	Use NDCG when ranked relevance matters, especially in scenarios where some documents are more relevant than others.
•	Example: Product search systems, where highly relevant results should appear at the top.
________________________________________









# Evaluation Metrics for Text Generation and RAG Systems
When evaluating the output of generative models (such as the generator in RAG pipelines), we often rely on metrics that measure the quality of the generated text. Below are some commonly used evaluation metrics:
________________________________________
## 1. BERTScore
What is BERTScore?
•	BERTScore uses contextual embeddings from pre-trained language models like BERT to evaluate the similarity between generated text and reference text.
•	Instead of relying on exact word matching, it computes token-level similarity using cosine similarity of embeddings.
How it Works?
1.	Compute contextual embeddings for tokens in both the reference and generated text using BERT.
2.	Match tokens in the generated text to tokens in the reference text based on maximum similarity.
3.	Calculate precision, recall, and F1-score based on these matches.
When to Choose This Metric?
•	Use BERTScore when semantic similarity is more important than exact word matching.
•	Example: Evaluating machine translations, paraphrases, or abstractive summaries where the generated text can vary significantly in phrasing.
Advantages:
•	Captures semantic meaning.
•	Robust to synonyms and word order changes.
________________________________________
## 2. BLEU (Bilingual Evaluation Understudy)
What is BLEU?
•	BLEU measures how closely a generated text matches one or more reference texts based on n-gram overlap.
•	Formula: BLEU=BP⋅exp⁡(∑n=1Nwn⋅log⁡pn)\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \cdot \log p_n \right)BLEU=BP⋅exp(n=1∑Nwn⋅logpn) where pnp_npn is the precision of n-grams, and BP (brevity penalty) penalizes short sentences.
How it Works?
•	BLEU counts matching n-grams between the generated and reference text.
•	It computes a weighted average of n-gram precision, with an optional penalty for brevity.
When to Choose This Metric?
•	Use BLEU for tasks where exact word matches are critical, such as machine translation or code generation.
•	Example: Comparing generated translations against gold-standard translations.
Limitations:
•	Penalizes valid paraphrasing.
•	Doesn't account for synonyms or word order.
________________________________________
## 3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
What is ROUGE?
•	ROUGE evaluates the quality of summaries by measuring the overlap of n-grams, word sequences, or word pairs between the generated text and reference text.
•	Common variants:
o	ROUGE-N: Measures overlap of n-grams (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).
o	ROUGE-L: Measures the longest common subsequence (LCS) between generated and reference texts.
How it Works?
•	ROUGE focuses on recall: how much of the reference text is captured by the generated text.
•	It compares the overlap of n-grams or sequences.
When to Choose This Metric?
•	Use ROUGE for tasks involving summarization or text generation where coverage of reference content is key.
•	Example: Evaluating abstractive or extractive text summarization models.
Advantages:
•	Easy to compute and interpret.
•	Focuses on recall, which is often more important in summarization tasks.
________________________________________
## 4. RAGAS
What is RAGAS?
•	RAGAS (Retriever-Augmented Generation Analysis Score) is a comprehensive framework for evaluating RAG systems. It measures both the retriever's and generator's performance together.
•	RAGAS focuses on retrieval quality, generated text quality, and factual consistency.
How it Works?
•	Combines traditional metrics for retrieval (e.g., Recall@K, Precision@K) with generation metrics (e.g., BLEU, ROUGE) and factual alignment checks.
•	Includes scoring for factual consistency between the retrieved documents and generated text using LLM-based scoring.
When to Choose This Metric?
•	Use RAGAS for end-to-end evaluation of RAG systems, where the quality of both the retriever and generator must be measured together.
•	Example: Evaluating a chatbot or QA system using RAG.
Advantages:
•	Holistic evaluation for RAG.
•	Ensures the generated output aligns with retrieved evidence.
________________________________________
## 5. deepeval
What is deepeval?
•	deepeval is a deep learning-based evaluation framework for text generation tasks. It leverages neural models to assess the quality of generated text based on contextual embeddings, coherence, fluency, and semantic similarity.
How it Works?
1.	Uses pre-trained or fine-tuned deep learning models to evaluate generated text against reference text.
2.	Scores are based on semantic similarity, coherence, fluency, and relevance.
When to Choose This Metric?
•	Use deepeval for advanced evaluations that go beyond simple n-gram overlaps, especially for open-ended text generation tasks.
•	Example: Evaluating chatbots, story generation, or creative writing models.
Advantages:
•	Captures nuanced aspects of text quality.
•	Works well for open-ended generation tasks.

