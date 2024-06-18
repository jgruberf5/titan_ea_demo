## What's This Going To Do?

When you <span style="font-size: 1.2em; font-style: italic; color: lightblue;">hit the 'Start Training' button</span>, 
the agent application will go fetch the list of URLs from the *BIG-IP Next SPK* documentation. 

<div style="text-align:center;">
<img src="file/assets/f5_rag_app_training_egress.png" alt="F5 SPK Provides Egress for AI training" style="width:200px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

It will then:

- tokenize the text
- create vector embeddings from the text tokens local embedding model
- create a vector database for semantic search of the text tokens
- create a keyword index for keyword search of the local documents

This phase will require egress networking to fetch the documents and to create the vector database.

Once we have all these, if you go back to the 
<span style="font-size: 1.2em; font-style: italic; color: lightblue;">'Ask Me a Question' tab and ask your question again</span>,
see if the answer comes from the local documents.