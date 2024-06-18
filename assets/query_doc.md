## Intelligent RAG Agent Applications in AI Clusters

Our RAG agent AI application is designed to use the AI cluster LLM, keyword search, and semantic search of locally 
trained documents to answer a question provided by the client correctly. If no locally trained document has an answer
it will do an Internet web search and answer from what it finds there. 

<div style="text-align:center;">
<img src="file/assets/f5_client_ingress_to_inferencing.png" alt="F5 SPK Provides Ingress to our RAG application" style="width:200px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

Let's start by asking our agent application a question before we have training it on any local documents. 

<span style="font-size: 1.2em; font-style: italic; color: lightblue;">Type 'What is SPK?' in the question field and hit Enter.</span>

*Your browser client required BIG-IP Next SPK ingress services to place the query API request to our RAG agent application. If an ingress
service had not been defined, the BIG-IP Next SPK default deny firewall would have denied access to our RAG agent application.*

Because its keyword search is empty, our RAG agent application should immediately try to retrieve documents from the Internet via web search. 

*If BIG-IP Next SPK egress had not been defined, our secured RAG agent application service would not any proper path to the Internet to 
perform the web query. Because BIG-IP Next SPK put our Kubernetes tenant in the right egress network such that upstream data center security 
could properly identify it and allow access for our web search.*

<div style="text-align:center;">
<img src="file/assets/f5_rag_app_egress.png" alt="F5 SPK Provides Egress for External Services Access" style="width:200px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

If we had trained on local documents, then the agent would use the LLM to see if the question matched the keywords (keyword search). 
If the local documents were relevant, a semantic search of a vector database is done to get only the relevant language tokens (fragments) 
which are used for the LLM to generate your answer. 

Our RAG agent application asked the LLM to make sure its generated answer came from the local documents and ask it again to make sure
its generated answer actually answered the question.  In all the RAG agent application can make many repeated queries of the local 
AI cluster LLM to see how best to answer your question. Taking multiple passes through an local LLM  is common for self-correcting 
RAG agent applications which use the LLM's own language predicting ability to improve accuracy and provide
valid answers from data it was never trained on.

Now, let's train on some local documents, then we will ask the same question again and see the answer change! 

<span style="font-size: 1.2em; font-style: italic; color: lightblue;">Click on the 'RAG Corpus Training' tab and create a local corpus of knowledge for our agent.</span>