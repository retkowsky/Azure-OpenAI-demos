![image](https://github.com/Azure/aoai-apim/assets/9942991/19804cb2-ec27-4a20-914c-166826b31194)

# Azure OpenAI Using PTUs/TPMs With API Management     	- Using the Scaling Special Sauce

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

# Scaling (Single Region)
There are other articles/repos which describe this basic scenario, and also provide configurations for the basic APIM setup used with AOAI, so we will not re-invent the wheel here. Examples are in the Reference section near the end of this article. 
However, it is important to note that you can create a "spillover" scenario, where you may be sending prompts to PTUs enabled deployment of an AOAI account, and then if you exceed PTU limits, then send (or spillover) onto TPM enabled AOAI account, used in the pay-as-you-go.

Here is the basic setup, however this architecture can scale and also include many other resources, however for simplicity and focus, only relevant services are depicted here:
![image](https://github.com/Azure/aoai-apim/assets/9942991/896b69b8-5251-4ed6-a189-e887f9071515)

We will take this a step further to understand how to queue messages for AOAI, and also manage the rate limits and return code/error handling for the AOAI model deployments.

# The Scaling Special Sauce

So how do we control (or queue) messages when using multiple Azure OpenAI instances (accounts)? How do we manage return error codes highly efficiently to optimize the AOAI experience?

As a best practice, Microsoft recommends the use of **retry logic** whenever using a service such as AOAI.  With APIM, this will allow us do this easily, but with some secret sauce added it... using the concept of _retries with exponential backoff_.
Retries with exponential backoff is a technique that retries an operation, with an exponentially increasing wait time, up to a maximum retry count has been reached (the exponential backoff). This technique embraces the fact that cloud resources might intermittently be unavailable for more than a few seconds for any reason, or more likely using AOAI, if an error is returned due to too many tokens per second (or RPM) in a large scale deployment.

This can be accomplished via the [APIM Retry Policy](https://learn.microsoft.com/en-us/azure/api-management/retry-policy).

	<retry condition="@(context.Response.StatusCode == 429 || context.Response.StatusCode >= 500)" interval="1" delta="1" max-interval="30" count="13">

Note in the above example, the error is specific to an response status code equal to '429', which is the return code for 'server busy', which states too many concurrent requests were sent to the model, per second.

**And extremely important**: When the APIM **interval, max-interval AND delta** parameters are specified, then an **exponential interval retry algorithm** is automatically applied. This is the special sauce needed to scale.

Without this scaling special sauce (APIM using retries with exponential backoff), once the initial rate limit is hit, say due to many concurrent users sending too many prompts, then a '429' error return code (server busy) response code is sent back. As additional subsequent prompts/completions are being sent, then the issue can be compounded quickly as more 429 errors are returned, and the error rates increase further and further. 
It is with the retries with exponential backoff where you are then able to scale many thousands of concurrent users with very low error responses, providing scalability to the AOAI service. I will include some metrics in the near future on scaling 5K concurrent users with low latency and less than 0.02% error rate.
 
In addition to using restries with exponential backoff, Azure APIM also supports content based routing. Content based routing is where the message routing endpoint is determined by the **content** of the message at runtime. You can leverage this to send AOAI prompts to multiple AOAI accounts, including both PTUs and TPMs, for meeting further scaling requirements.
For example, if your model API request states a specific version, say gpt-35-turbo-16k, you can then route this request to your GPT 3.5 Turbo (16K) PTUs deployment. We won't get into too much details here, but there are additional repo examples in the references section at the end of this repo.

In the [infra](./infra/) directory you will find a sample Bicep template to deploy Azure APIM and an API that applies this exponential retry logic and optional failover between different Azure OpenAI deployments. You will need to have an Azure subscription and two Azure OpenAI LLM deployments. Once deployed, you will need to give the APIM's system assigned managed identity the role of Cognitive Services OpenAI User on the Azure OpenAI accounts it's connected to, and add any required networking configurations.

# Multi-Region

As described in the single-region scenario above, you can use APIM to queue and send prompts to any AOAI endpoint, as long as those endpoints can be reached. In a multi-region example below, we have two AOAI accounts in one region (one PTU and another TPM), and then a 3rd Azure OpenAI account in another Azure region.  

A single API Management service can easily scale and support many AOAI accounts, even across multiple regions.

![image](https://github.com/Azure/aoai-apim/assets/9942991/c44fa3bc-5980-4d21-900f-6f87c7716f1f)

Please take a look at the multi-region APIM best practices below (item #4) to understand when to use additional APIM instances.

# Best Practices

1. HTTP Return Codes/Errors  	 
	As described in the Special Sauce section above, you can use retries with exponential backoff for any 429 errors, based on the [APIM Retry Policy document](https://learn.microsoft.com/en-us/azure/api-management/retry-policy).
	However, as a best practice, you should always configure error checking on the size of prompt vs the model this prompt is intended for, first. For example, for GPT-4 (8k), this model supports a max request token limit of 8,192.  If your prompt is 10K in size, then this will fail, AND ALSO any subsequent retries would fail as well, as the token limit size was already exceeded.
	As a best practice, ensure the size of the prompt does not exceed the max request token limit immediately, prior to sending the prompt across the wire to the AOAI service. Again here are the [token size limits for each model](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models).
  
	This table describes **a few of the common** HTTP Response Codes for AOAI:
	HTTP Response Code | Cause | Remediation | Notes
	--- | --- | --- | ---
	200 | Processed the prompt. Completion without error | N/A |
	429 (v0613 AOAI Models)	|  Server Busy (Rate limit reached for requests) | APIM - Retries with Exponential Backoff |When APIM interval, max-interval and delta are 	specified, an exponential interval retry algorithm is automatically applied.
	424 (v0301 AOAI Models)	| Server Busy (Rate limit reached for requests) | APIM - Retries with Exponential Backoff | Same as above
	408  | Request timeout | APIM Retry with interval | Many reasons why a timeout could occur, such as a network connection/transient error.
	50x |	Internal server error due to transient error or backend AOAI internal error |	APIM Retry with interval| See Retry Policy Link below
	400 |	Other issue with the prompt, such as size too large for model type | Use APIM Logic or applicaiton logic to return custom error immediately | No further processing needed.

	**Retry Policy**: https://learn.microsoft.com/en-us/azure/api-management/retry-policy	
	
2. Auto-update to Default and Setting Default Models

	If you are still in the early testing phases for inference models, we recommend deploying models with the 'auto-update to default' set whenever it is available. When a new model version is introduced, you will want to ensure your applications and services are tested and working as expected against the latest version first. It is a best practice not to make newest model the DEFAULT until after successful testing and until the organization is ready to move to the newer model. After successful integration testing, you can make the latest model the default, which will then update the model deployment automatically within two weeks of a change in the default version.

	![image](https://github.com/Azure/aoai-apim/assets/9942991/ca44d6fc-4336-44d6-8e73-b8b71ade19fb)

3. Purchasing PTUs
	* Charges for PTUs are billed **up-front** for the entire month, starting on the day of purchase. The PTUs are not charged in arrears, that is, after the service has been used over the month period.
	* The month period does not necessarily fall exactly on the first day of month to the last day of each month, but instead when the PTUs were purchased. For example, if you purchased and applied the PTUs on the 9th day of the month, then you will be charged from the 9th until the following month, 8th day.
   	* As the term of the commitment is one month, PTUs can not be reduced. However, PTUs can be _added_ to a commitment mid-month.
   	* If a commitment is not renewed, deployed PTUs will be reverted to per hour pricing.
			
4. Multi-Region APIM Service

	Azure API Management has 3 _production_ level tiers - Basic, Standard, and Premium.
	The Premium tier enables you to distribute a single Azure API Management instance across any number of desired Azure regions. When you initially create an Azure API Management service, the instance contains only one unit and resides in a single Azure region (the primary region).

	What does this provide? If you have a multi-regional Azure OpenAI deployment, does this mean you are required to also have a multi-region (Premium) SKU of APIM? No, not necessarily. As you can see in the multi-region architecture above, a single APIM service instance can support multi-region, multi-AOAI accounts. Having a single APIM service makes sense when an application using the service is in the same region and you do not need DR.

	The Premium SKU gives you is the ability to have one region be the primary and any number of regions as secondaries. Then when should use a secondary, or secondaries?
	a. If you are planning for any DR scenarios, which is always recommended for any enterprise architecture. Note: Your enterprise applications should then also be designed for data resiliency, using DR strategies.
	b. As you are monitoring the APIM services, if you are seeing extremely heavy usage and are able to scale out your application(s) across regions, then you may want to deploy APIM service instances across multiple regions.

	For more information on [how to deploy an Azure API Management service instance to multiple Azure regions](https://learn.microsoft.com/en-us/azure/api-management/api-management-howto-deploy-multi-region).

### Additional Best Practices 
To minimize issues related to rate limits, use the following techniques:
* Set max_tokens and best_of to the minimum values that serve the needs of your scenario. For example, don’t set a large max-tokens value if you expect your responses to be small as this may increase response times.
* Use quota management to increase TPM on deployments with high traffic, and to reduce TPM on deployments with limited needs.
* Avoid sharp changes in the workload. Increase the workload gradually.
* Test different load increase patterns.

## References

- [APIM Retry Policy document](https://learn.microsoft.com/en-us/azure/api-management/retry-policy)
- [Enterprise Azure OpenAI Monitoring and Logging](https://github.com/Azure-Samples/openai-python-enterprise-logging)
- [Rate limit best practices](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/quota?tabs=rest#rate-limit-best-practices)
  
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
