# Azure OpenAI

## Introduction
While there are already a few articles and reference architectures available for using Azure OpenAI and Azure OpenAI Landing Zones, this article/repo focuses on AOAI + APIM, **deploying at scale** using PTUs (Reserved Capacity) and TPM (Pay-As-You-Go), and best practices around this.

### Brief Review of AOAI and APIM

**Azure OpenAI (AOAI)**: Azure OpenAI Service provides generative AI technology for all using REST API access to OpenAI's powerful language models such as GPT4, GPT3.5 Turbo, Embeddings model series and others. By now, you should be already be familiar with the [Azure OpenAI service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) so we won't go into those details.

**API Management (APIM)**: APIs are the foundation of an API Management service instance. Each API represents a set of operations available to app developers.
Each API contains a reference to the backend service that implements the API, and its operations map to backend operations. 
Operations in API Management are highly configurable, with control over URL mapping, query and path parameters, request and response content, and operation response caching. You can read [additional details on using APIM](https://learn.microsoft.com/en-us/azure/api-management/api-management-key-concepts).
In this article, we will cover configurations for APIM against AOAI service, and scaling this.

Azure OpenAI provides an API endpoint to consume the AOAI service, and APIM utilizes this AOAI endpoint.
Using APIM with AOAI, you can manage and implement policies to allow queuing, rate throttling, error handling, and managing usage quotas.
When using Azure OpenAI with API Management, this gives you the most flexibility in terms of both queuing prompts (text sent to AOAI) as well as return code/error handling management. More later in this document on using APIM with AOAI.

## Understanding TPMs, RPMs and PTUs
First, let's define TPMs, RPMs and PTUs in this section.
As we continue to understand scaling of the Azure OpenAI service, the Azure OpenAI's quota management feature enables assignment of rate limits to your deployments. It is important to remember that TPM's and PTUs are both rate limits AND are also used for billing purposes.

### TPMs
Azure OpenAI's quota management feature enables assignment of rate limits to your deployments, up-to a global limit called your “quota”. Quota is assigned to your subscription on a per-region, per-model basis in units of Tokens-per-Minute (TPM), by default. The billing component of TPMs is also known as pay-as-you-go, where pricing will be based on the pay-as-you-go consumption model, with a price per unit for each model. 

When you onboard a subscription to Azure OpenAI, you'll receive default quota for most available models. Then, you'll assign TPM to each deployment as it is created, and the available quota for that model will be reduced by that amount. 
TPMs/Pay-as-you-go are also the **default** mechanism for billing the AOAI service. 
Our focus for this article is not billing/pricing, but you can learn more about the [AOAI quota managment](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/quota?tabs=rest) or [Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/).

### RPM
A Requests-Per-Minute (RPM) rate limit will also be enforced whose value is set proportionally to the TPM assignment using the following ratio:
6 RPM per 1000 TPM

RPM is not a billing component directly, however it is a component of rate limits. It is important to note that while the billing for AOAI service is token-based (TPM), the actual two triggers which rate limits occur are as follows:
1) On a per **second** basis, not at the per **minute** billing level. And,
2) The rate limit will occur at either TPS (tokens-per-second) or RPM evaluated over a small period of time (1-10 seconds). That is, if you exceed the total tokens per second for a specific model, then a rate limit applies. If you exceed the RPM over a short time period, then a rate limit will also apply, returning limit error codes (429).

The throttled rate limits can easily be managed using the scaling special sauce, as well as following some of the best practices described later in this document.

You can read more about [quota management and the details on how TPM/RPM rate limits apply](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/quota?tabs=rest).

### PTUs 
Microsoft recently introduced the ability to use reserved capacity, or Provisioned Throughput Units (PTU), for AOAI earlier this summer.
Beyond the default TPMs described above, this new Azure OpenAI service feature, PTUs, defines the model processing capacity, **using reserved resources**, for processing prompts and generating completions.  

PTUs are purchased as a monthly commitment with an auto-renewal option, which will RESERVE AOAI capacity within an Azure subscription, using a specific model, in a specific Azure region. 
Let's say if you have 300 PTUs provisioned for GPT 3.5 Turbo, the PTUs are provisioned for only GPT 3.5 Turbo deployments, within a specific Azure subscription, and not for GPT 4. You can have separate PTUs for GPT 4, with minimum PTUs described in the table below.  

Keep in mind, while having reserved capacity does provide consistent latency and througput, throughput is highly dependent on your scenario. Throughput will be affected by a few items including number and ratio of prompts and generation tokens, number of simultaneous requests, and the type and version of model used.

Table describing approximate TPMs expected in relation to PTUs, per model.

![image](https://github.com/Azure/aoai-apim/assets/9942991/b24f5193-92cc-4cef-af73-e172b9ad1b73)


## Limits
As organizations scale using Azure OpenAI, as described above, there are rate **limits** on how fast tokens are processed, in the prompt+completion request. There is a limit to how much text prompts can be sent due to these token limits for each model that can be consumed in a single request+response. 
It is important to note the overall size of tokens for rate limiting include BOTH the prompt (text sent to the AOAI model) size PLUS the return completion (response back from the model) size, and also this token limit varies for each different AOIA model type. 

For example,  with a quota of 240,000 TPM for GPT-35-Turbo in Azure East US region, you can have a single deployment of 240K TPM, 2 deployments of 120K TPM each, or any number of deployments in one or multiple deployments as long as the TPMs add up to 240K (or less) total in that region in East US.

As our customers are scaling, they can [add an additional Azure OpenAI accounts](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).

The maximum Azure OpenAI resources per region per Azure subscription is 30 (at the time of this writing) and also dependent on regional capacity **availability.** 
