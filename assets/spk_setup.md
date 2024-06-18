# Ingress and Egress Protection of Inferencing Applications - Orchestrated through Kubernetes

Inferencing applications, like this RAG agent applications, get installed inside AI clusters so that the LLM's they utilize for
natural language response generation can take advantage of the parallel processing resources. *BIG-IP Next SPK* provides a means to 
orchestrate and intelligently get client traffic into the AI cluster, and it provides a means for workflows deployed within 
the cluster to intelligently route traffic to external services.

## What's Already Been Provisions through BIG-IP Next SPK

<div style="text-align:center;">
<img src="file/assets/f5_spk_on_bf3_dpu.png" alt="F5 SPK on NVIDIA BlueField 3" style="width:512px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

Our SRE team installed an AI Factory cluster built on an *NVIDIA* MGX chassis. The installation includes *NVIDIA* BlueField 3 DPUs to provide networking for the cluster. *BIG-IP Next SPK* was deployed, using Kubernetes, to the BlueField 3 DPU ARM complex as a *whole cluster SPK*.

Through Kubernetes CRDs (customer resource definitions), BGP peering with our data center infrastructure was declared. The *BIG-IP next SPK* controller orchestrated the 
dynamic routing configuration defined in the CRDs. Now our AI Factory cluster has joined our data center network fabric. 

```
tmm:
  dynamicRouting:
    enabled: true
    tmmRouting:
      config:
        bgp:
          asn: 100
          hostname: spk-bgp
          neighbors:
          - ip: 10.10.10.200
            asn: 200
            ...
```


After *BIG-IP Next SPK* was connected to our networking fabric, our NetOps team defined *F5SPKVlan* Kubernetes resources defining network ingress and egress for our AI cluster tenant.

Our AI cluster tenant now has a means to have its networking traffic identified within the data center network. Having a means to orchestrate an maintain network segmentation *per tenant* is the basis not only for monitoring and data center network security, but is also the means by which network service levels are maintained. *BIG-IP Next SPK* is the *glue* between the Kubernetes *namespace* based workload segmentation and the data center network services segmentation.

<div style="text-align:center;">
<img src="file/assets/f5_spk_ingress_egress_vlans.png" alt="F5 SPK Provides Network Segregation" style="width:400px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

```
---
apiVersion: "k8s.f5net.com/v1"
kind: F5SPKVlan
metadata:
  name: "vlan-external"
spec:
  name: external
  tag: 3334
  interfaces:
    - "1.1"
  selfip_v6s:
    - 2002::198:51:100:120
  prefixlen_v6: 112
---
apiVersion: "k8s.f5net.com/v1"
kind: F5SPKVlan
metadata:
  name: "vlan-internal"
spec:
  name: internal
  tag: 3335
  interfaces:
    - "1.2"
  selfip_v6s:
    - 2002::198:51:200:120
  prefixlen_v6: 112
```

Then our DevOps team deployed this agent application from a CI/CD pipeline which declared the application along with its Kubernetes services it supplies by name. 

<div style="text-align:center;">
<img src="file/assets/f5_spk_ingress_vs.png" alt="F5 SPK Provides Firewalling and Load Balancing" style="width:256px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>

An appropriate *F5SPKIngressTCP* is declared to delivery client traffic to our application.

```
---
apiVersion: ingresstcp.k8s.f5net.com/v1
kind: F5SPKIngressTCP
metadata:
  name: app-ingress-example
service:
  name: k8s-service
  port: 8050
spec:
  clientTimeout: 30
  destinationAddress: 10.11.22.33
  ipv6destinationAddress: 2002:0007::0008
  destinationPort: 8050
```

An appropriate *F5SPKSnatpool* and *F5SPKEgress* are defined, giving our AI agent application a way to make requests out of the AI cluster to external services.
Controlling the egress and network translation of tenant workload traffic outbound towards your data center networking creates basis for monitoring, 
throttling, and securing outbound communications from your AI clusters to the world.

<div style="text-align:center;">
<img src="file/assets/f5_spk_egress_snat.png" alt="F5 SPK Provides Egress Network Address Translation and Security" style="width:256px; vertical-align: middle; float:right; margin: 20px 20px 20px 20px;"/>
</div>


```
apiVersion: k8s.f5net.com/v1
kind: F5SPKSnatpool
metadata:
  name: egress-snatpool
spec:
  name: snatpool1
  addressList:
  - - 10.200.1.1
    - 10.201.0.1
  - - 10.200.0.2
    - 10.201.0.2
  - - 10.200.0.3
    - 10.201.0.3
  - - 10.200.0.4
    - 10.201.0.4
---
apiVersion: k8s.f5net.com/v2
kind: F5SPKEgress
metadata:
  name: egress
spec:
  debugLogEnabled: false
  dnsCacheName: ""
  dnsNat46Enabled: false
  dnsNat46Ipv4Subnet: 10.2.2.0/24
  dnsNat46SorryIp: 192.168.1.1
  dnsRateLimit: 0
  dualStackEnabled: false
  egressSnatpool: snatpool1
  maxDNS46TTL: 120
  maxReservedStaticIps: 0
  maxTmmReplicas: 1
  nat64Enabled: false
  nat64Ipv6Subnet: 64:ff9b::/96
```

Now that we are deployed, it's time to use our agent application in action. 

<span style="font-size: 1.2em; font-style: italic; color: lightblue;">Select the 'Ask Me a Question' tab.</span>