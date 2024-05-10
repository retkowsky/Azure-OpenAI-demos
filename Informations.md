# Azure OpenAI

## Quotas and limits
https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits

## Models
https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models

## Pricing
https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

## What's new
https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new

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

https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput


## Limits
As organizations scale using Azure OpenAI, as described above, there are rate **limits** on how fast tokens are processed, in the prompt+completion request. There is a limit to how much text prompts can be sent due to these token limits for each model that can be consumed in a single request+response. 
It is important to note the overall size of tokens for rate limiting include BOTH the prompt (text sent to the AOAI model) size PLUS the return completion (response back from the model) size, and also this token limit varies for each different AOIA model type. 

For example,  with a quota of 240,000 TPM for GPT-35-Turbo in Azure East US region, you can have a single deployment of 240K TPM, 2 deployments of 120K TPM each, or any number of deployments in one or multiple deployments as long as the TPMs add up to 240K (or less) total in that region in East US.

As our customers are scaling, they can [add an additional Azure OpenAI accounts](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).

The maximum Azure OpenAI resources per region per Azure subscription is 30 (at the time of this writing) and also dependent on regional capacity **availability.** 
