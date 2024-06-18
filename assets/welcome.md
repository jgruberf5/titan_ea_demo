
<div style="text-align:center;">
<img src="file/assets/f5_k8s_ai_icons.png" alt="F5 SPK Secures Access to K8s AI-Enhanced Applications" style="width:128px; vertical-align: middle; float:left; margin: 10px 10px 10px 10px;"/>
</div>

# BIG-IP Next Service Proxy for Kubernetes (SPK) for AI Applications

The rapid proliferation of AI-enhanced applications is revolutionizing industries worldwide, transforming how businesses operate and how consumers engage with technology. As AI continues to evolve, its integration into everyday applications promises to unlock unprecedented opportunities, fostering a smarter, more connected world where possibilities are limited only by our imagination.

AI-enhanced applications leverage Kubernetes for its robust orchestration capabilities, ensuring efficient deployment, scaling, and management of complex AI workloads. Its scalability is crucial for AI applications, which often require significant computational resources to process large datasets and perform real-time analytics. These computations are optimized by running teams of parallel operations on stream processors found in graphical processing units (GPUs).  Kubernetes enhances resource utilization of these precious compute resources by dynamically allocating them based on demand, optimizing cost-efficiency.

While typically the frontend user interface would not be run inside of the AI clusters hosting the expensive GPUs, this interface is provided to help you follow along with what exactly goes in (ingress) and out (egress) of the AI cluster to support AI-enhanced applications.

<div style="text-align:center;">
<img src="file/assets/ai_inferencing_and_answer.png" alt="F5 SPK Secures Access to K8s AI-Enhanced Applications" style="width:256px; vertical-align: middle; float:right; margin: 10px 10px 10px 10px;"/>
</div>

This application is a micro-services based AI Retrieval Augmented Generation (RAG) agent that utilizes the power of AI to generate natural language responses to end users queries. This is called an *inferencing workflow* because the outcome is a generated document that is created based on an AI model inferred language output which matches the query.

Our agent process running in the AI cluster will route queries made by applications to a set of documents which can best answer their query. The locally hosted LLM    will take advantage of our available GPUs to not just generate our response, but to grade itself on how well it did answering you query.

<div style="text-align:center;">
<img src="file/assets/ai_training_rag_embedding.png" alt="F5 SPK Secures Access to K8s AI-Enhanced Applications" style="width:256px; vertical-align: middle; float:right; margin: 10px 10px 10px 10px;"/>
</div>

Our RAG agent application will have two sources of documents and will decide, based on your query, which set of documents the LLM uses to generate your answer. One choice of documents will be a set of locally *trained* documents which have been indexed for traditional keyword search and tokenized, then vector embedded in to a vector database, for semantic search based on language similarity to the question. The ingestion, indexing, tokenization, and embedding vectorization of the local documents is a *training* workflow for our RAG application which takes advantage of our GPUs in the cluster as well.

## What Does BIG-IP Next Service Proxy for Kubernetes do for AI Applications?

<div style="text-align:center;">
<img src="file/assets/f5_spk_for_ai.png" alt="F5 SPK Secures Access to K8s AI-Enhanced Applications" style="width:500px; vertical-align: middle; float:right; margin: 10px 10px 10px 10px;"/>
</div>

<span style="font-size: 1.2em; font-style: italic; color: lightblue;">How can service providers and enterprises deploy, protect, and scale AI-enhanced applications in Kubernetes clusters without creating whole new operational processes to support their use?</span>

<span style="font-size: 1.2em; font-style: italic; color: lightblue;">How was this application exposed to the real world securely from inside the AI Kubernetes cluster?</span>


F5 has always been the industry standard for application delivery and security. Our high-performance full-proxy architecture is the gold standard for true segmented security and service availability.

What many do not know is that we brought the same high-performance full-proxy engine into the modern application world and orchestrated it into the heart of Kubernetes with *BIG-IP Next Service Proxy for Kubernetes*.
*BIG-IP Next Service Proxy for Kubernetes* brings ingress and egress control of all North-South traffic for Kubernetes clusters, including AI clusters. *BIG-IP Next Service Proxy for Kubernetes* is controlled through 
Kubernetes through the use of common and custom Kubernetes resource declarations. 

*BIG-IP Next Service Proxy for Kubernetes* also maps native Kubernetes resource namespaces to segregated data center network traffic flows for tenant VLANs, encapsulated overlay networks, or IP subnets. This effectively maps
Kubernetes tenancy to data center network tenancy, thus enabling the extension of existing network infrastructure seamlessly into AI clusters.

<span style="font-style: italic; color: lightblue;">DevOps gets what they want - orchestration through Kubernetes</span>

<span style="font-style: italic; color: lightblue;">NetOps gets what they want - seamless extension of network controls, monitoring, and security</span>

<span style="font-style: italic; color: lightblue;">MLOps and Data Science get what they want - access to local AI cluster resources</span>

