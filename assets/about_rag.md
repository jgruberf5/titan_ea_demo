# <span style="color:lightblue">R.A.G. (Retrieval Augmented Generation)</span>
LLM are trained on vast amounts of language examples to be able to produce a predictive natural language outcome to prompted inputs.

However, don't ask an LLM to produce a natural language response based on any proprietary language or vocabulary which was not in their training data.

How can we take advantage of the natural language generation capabilities of LLMs, but augment their training data with a corpus of our own proprietary knowledge? That is where R.A.G. comes in. 

## <span style="color:lightblue">Training in R.A.G.</span>

LLM processing requirements and speed of response are highly dependent on the complexity of the prompted request made of them. While you could ask the LLM to generate a response based on the whole of your corpus of data, not only would that not yield the response accuracy you are looking for, but it would also heat the planet for extended periods of time. All around a bad idea.

So how do we narrow down a corpus of information to just related language? We have been using search engines to do that for years. First, we chop down our content into meaningful, but manageable chucks, or tokens, which can be processed quickly by parallel processes, like the ones GPU stream processor are so good at doing quickly. We can do both indexed keywords searching, which is good at narrowing down our content to specific words, as well as semantic search based on numerical analysis of the language in our corpus. Keyword extraction is pretty simply. To do the semantic search we need to turn our tokenized text into numerical embedded vectors which can be stored and searched in a vector database. We can then search the vector database for all the text that is highly similar semantically to the prompt question being asked.

Combining our LLM which generates natural language response with a searchable pile of corpus data to base a response we can create an inferencing workflows which will give us what we want.

## <span style="color:lightblue">Inferencing in R.A.G.</span>

In this apps corrective R.A.G. workflow, when a question is received, we can first use our LLM to ask if the question being asked is related to our pile of keywords. That will give us a quick and dirty way to decide if the question is related to our corpus at all. If the question is not related, we just do a web search for documents and get a temporary set of documents to use to generate the response.

If we determine that the question is related to our corpus, we quickly use the same embedding model which turned our corpus tokens into numerical vectors and then query our vector database to return only a few documents related to our question which we will in our prompt to get the LLM to generate a natural language response. We tell the LLM to answer the question based on this highly reduced set of document tokens the vector database told us were related to our question.

But what about those reports of LLM predicting language that just 'looses its mind' and makes no sense to our context. Those are called LLM hallucinations. We will put two corrective measures in place in our workflow to assist with that problem. 

First, we will take another pass at our LLM to assure that the generated response is 'grounded' in the corpus of the relevant text we supplied. We do that with some fancy 'prompt engineering'. If the generated is not 'grounded', then the LLM lost its mind. That is the first thing we do.. get grounded. (I was always grounded as a kid, so I'm used to it!)

Second, we will ask the LLM if the answer it supplied actually answers the question. Again, just some fancy prompt work. This is a summarization type task that LLMs are really good at doing. 

Once we:
    -	Have a set of relevant document tokens from our corpus or the web
    -	Asked the LLM to answer the question from that set of relevant document tokens
    -	Asked the LLM if the generated response is 'grounded' in the document tokens
    -	Asked the LLM if the generated response answered the question

We can confidently return a generated response to the prompted question. If not, we can tell the LLM to generate another response a few times to see if it can do any better. Remember, the generated response has some creativity and randomness built into the LLM, just like our real language does. 

So we:

[x] **Retrieved** a small relevant set of document tokens related to our question. In our case we use a corpus of documents tokenized, keyword indexed, and semantically searched, or else a web search.

[x] **Augmented**  the LLM data by telling it to answer from the retrieved relevant data.

[x] **Generated**  an answer with the LLM that is validated from our relevant data and actually answers the prompted question.

There you have it -> **R.A.G.**

*Titan AI PM Team*