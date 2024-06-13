import * as fs from 'fs';
import fetch from 'node-fetch';

import { AwsBedrockCompletionProvider } from '../src/providers/bedrock';
import {
  OpenAiAssistantProvider,
  OpenAiCompletionProvider,
  OpenAiChatCompletionProvider,
} from '../src/providers/openai';
import { AnthropicCompletionProvider } from '../src/providers/anthropic';
import { LlamaProvider } from '../src/providers/llama';

import { clearCache, disableCache, enableCache } from '../src/cache';
import { loadApiProvider, loadApiProviders } from '../src/providers';
import {
  AzureOpenAiChatCompletionProvider,
  AzureOpenAiCompletionProvider,
} from '../src/providers/azureopenai';
import { OllamaChatProvider, OllamaCompletionProvider } from '../src/providers/ollama';
import { WebhookProvider } from '../src/providers/webhook';
import {
  HuggingfaceTextGenerationProvider,
  HuggingfaceFeatureExtractionProvider,
  HuggingfaceTextClassificationProvider,
} from '../src/providers/huggingface';
import {
  CloudflareAiChatCompletionProvider,
  CloudflareAiCompletionProvider,
  CloudflareAiEmbeddingProvider,
} from '../src/providers/cloudflare-ai';

import type { ProviderOptionsMap, ProviderFunction } from '../src/types';

jest.mock('fs', () => ({
  readFileSync: jest.fn(),
  writeFileSync: jest.fn(),
  statSync: jest.fn(),
  readdirSync: jest.fn(),
  existsSync: jest.fn(),
  mkdirSync: jest.fn(),
  promises: {
    readFile: jest.fn(),
  },
}));

jest.mock('glob', () => ({
  globSync: jest.fn(),
}));

jest.mock('node-fetch', () => jest.fn());
jest.mock('proxy-agent', () => ({
  ProxyAgent: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../src/esm');

jest.mock('fs', () => ({
  readFileSync: jest.fn(),
  existsSync: jest.fn(),
  mkdirSync: jest.fn(),
}));

jest.mock('glob', () => ({
  globSync: jest.fn(),
}));

jest.mock('../src/database');

describe('call provider apis', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('OpenAiCompletionProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          choices: [{ text: 'Test output' }],
          usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
        }),
      ),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OpenAiCompletionProvider('text-davinci-003');
    const result = await provider.callApi('Test prompt');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
  });

  test('OpenAiChatCompletionProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          choices: [{ message: { content: 'Test output' } }],
          usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
        }),
      ),
      ok: true,
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
    const result = await provider.callApi(
      JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
    );

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
  });

  test('OpenAiChatCompletionProvider callApi with caching', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          choices: [{ message: { content: 'Test output 2' } }],
          usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
        }),
      ),
      ok: true,
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
    const result = await provider.callApi(
      JSON.stringify([{ role: 'user', content: 'Test prompt 2' }]),
    );

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output 2');
    expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

    const result2 = await provider.callApi(
      JSON.stringify([{ role: 'user', content: 'Test prompt 2' }]),
    );

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result2.output).toBe('Test output 2');
    expect(result2.tokenUsage).toEqual({ total: 10, cached: 10 });
  });

  test('OpenAiChatCompletionProvider callApi with cache disabled', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          choices: [{ message: { content: 'Test output' } }],
          usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
        }),
      ),
      ok: true,
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
    const result = await provider.callApi(
      JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
    );

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

    disableCache();

    const result2 = await provider.callApi(
      JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
    );

    expect(fetch).toHaveBeenCalledTimes(2);
    expect(result2.output).toBe('Test output');
    expect(result2.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

    enableCache();
  });

  test('OpenAiChatCompletionProvider constructor with config', async () => {
    const config = {
      temperature: 3.1415926,
      max_tokens: 201,
    };
    const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo', { config });
    const prompt = 'Test prompt';
    await provider.callApi(prompt);

    expect(fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        body: expect.stringMatching(`temperature\":3.1415926`),
      }),
    );
    expect(provider.config.temperature).toBe(config.temperature);
    expect(provider.config.max_tokens).toBe(config.max_tokens);
  });
});

xdescribe('loadApiProvider', () => {
  test('loadApiProvider with filepath', async () => {
    const mockYamlContent = `id: 'openai:gpt-4'
config:
  key: 'value'`;
    (fs.readFileSync as jest.Mock).mockReturnValueOnce(mockYamlContent);

    const provider = await loadApiProvider('file://path/to/mock-provider-file.yaml');
    expect(provider.id()).toBe('openai:gpt-4');
    expect(fs.readFileSync).toHaveBeenCalledTimes(1);
    expect(fs.readFileSync).toHaveBeenCalledWith('path/to/mock-provider-file.yaml', 'utf8');
  });

  test('loadApiProvider with openai:chat', async () => {
    const provider = await loadApiProvider('openai:chat');
    expect(provider).toBeInstanceOf(OpenAiChatCompletionProvider);
  });

  test('loadApiProvider with openai:completion', async () => {
    const provider = await loadApiProvider('openai:completion');
    expect(provider).toBeInstanceOf(OpenAiCompletionProvider);
  });

  test('loadApiProvider with openai:assistant', async () => {
    const provider = await loadApiProvider('openai:assistant:foobar');
    expect(provider).toBeInstanceOf(OpenAiAssistantProvider);
  });

  test('loadApiProvider with openai:chat:modelName', async () => {
    const provider = await loadApiProvider('openai:chat:gpt-3.5-turbo');
    expect(provider).toBeInstanceOf(OpenAiChatCompletionProvider);
  });

  test('loadApiProvider with openai:completion:modelName', async () => {
    const provider = await loadApiProvider('openai:completion:text-davinci-003');
    expect(provider).toBeInstanceOf(OpenAiCompletionProvider);
  });

  test('loadApiProvider with OpenAI finetuned model', async () => {
    const provider = await loadApiProvider('openai:chat:ft:gpt-3.5-turbo-0613:company-name::ID:');
    expect(provider).toBeInstanceOf(OpenAiChatCompletionProvider);
    expect(provider.id()).toBe('openai:ft:gpt-3.5-turbo-0613:company-name::ID:');
  });

  test('loadApiProvider with azureopenai:completion:modelName', async () => {
    const provider = await loadApiProvider('azureopenai:completion:text-davinci-003');
    expect(provider).toBeInstanceOf(AzureOpenAiCompletionProvider);
  });

  test('loadApiProvider with azureopenai:chat:modelName', async () => {
    const provider = await loadApiProvider('azureopenai:chat:gpt-3.5-turbo');
    expect(provider).toBeInstanceOf(AzureOpenAiChatCompletionProvider);
  });

  test('loadApiProvider with anthropic:completion', async () => {
    const provider = await loadApiProvider('anthropic:completion');
    expect(provider).toBeInstanceOf(AnthropicCompletionProvider);
  });

  test('loadApiProvider with anthropic:completion:modelName', async () => {
    const provider = await loadApiProvider('anthropic:completion:claude-1');
    expect(provider).toBeInstanceOf(AnthropicCompletionProvider);
  });

  test('loadApiProvider with ollama:modelName', async () => {
    const provider = await loadApiProvider('ollama:llama2:13b');
    expect(provider).toBeInstanceOf(OllamaCompletionProvider);
    expect(provider.id()).toBe('ollama:completion:llama2:13b');
  });

  test('loadApiProvider with ollama:completion:modelName', async () => {
    const provider = await loadApiProvider('ollama:completion:llama2:13b');
    expect(provider).toBeInstanceOf(OllamaCompletionProvider);
    expect(provider.id()).toBe('ollama:completion:llama2:13b');
  });

  test('loadApiProvider with ollama:chat:modelName', async () => {
    const provider = await loadApiProvider('ollama:chat:llama2:13b');
    expect(provider).toBeInstanceOf(OllamaChatProvider);
    expect(provider.id()).toBe('ollama:chat:llama2:13b');
  });

  test('loadApiProvider with llama:modelName', async () => {
    const provider = await loadApiProvider('llama');
    expect(provider).toBeInstanceOf(LlamaProvider);
  });

  test('loadApiProvider with webhook', async () => {
    const provider = await loadApiProvider('webhook:http://example.com/webhook');
    expect(provider).toBeInstanceOf(WebhookProvider);
  });

  test('loadApiProvider with huggingface:text-generation', async () => {
    const provider = await loadApiProvider('huggingface:text-generation:foobar/baz');
    expect(provider).toBeInstanceOf(HuggingfaceTextGenerationProvider);
  });

  test('loadApiProvider with huggingface:feature-extraction', async () => {
    const provider = await loadApiProvider('huggingface:feature-extraction:foobar/baz');
    expect(provider).toBeInstanceOf(HuggingfaceFeatureExtractionProvider);
  });

  test('loadApiProvider with huggingface:text-classification', async () => {
    const provider = await loadApiProvider('huggingface:text-classification:foobar/baz');
    expect(provider).toBeInstanceOf(HuggingfaceTextClassificationProvider);
  });

  test('loadApiProvider with hf:text-classification', async () => {
    const provider = await loadApiProvider('hf:text-classification:foobar/baz');
    expect(provider).toBeInstanceOf(HuggingfaceTextClassificationProvider);
  });

  test('loadApiProvider with bedrock:completion', async () => {
    const provider = await loadApiProvider('bedrock:completion:anthropic.claude-v2:1');
    expect(provider).toBeInstanceOf(AwsBedrockCompletionProvider);
  });

  test('loadApiProvider with openrouter', async () => {
    const provider = await loadApiProvider('openrouter:mistralai/mistral-medium');
    expect(provider).toBeInstanceOf(OpenAiChatCompletionProvider);
    // Intentionally openai, because it's just a wrapper around openai
    expect(provider.id()).toBe('mistralai/mistral-medium');
  });

  test('loadApiProvider with cloudflare-ai', async () => {
    const supportedModelTypes = [
      { modelType: 'chat', providerKlass: CloudflareAiChatCompletionProvider },
      { modelType: 'embedding', providerKlass: CloudflareAiEmbeddingProvider },
      { modelType: 'embeddings', providerKlass: CloudflareAiEmbeddingProvider },
      { modelType: 'completion', providerKlass: CloudflareAiCompletionProvider },
    ] as const;
    const unsupportedModelTypes = ['assistant'] as const;
    const modelName = 'mistralai/mistral-medium';

    // Without any model type should throw an error
    await expect(() => loadApiProvider(`cloudflare-ai:${modelName}`)).rejects.toThrowError(
      /Unknown Cloudflare AI model type/,
    );

    for (const unsupportedModelType of unsupportedModelTypes) {
      await expect(() =>
        loadApiProvider(`cloudflare-ai:${unsupportedModelType}:${modelName}`),
      ).rejects.toThrowError(/Unknown Cloudflare AI model type/);
    }

    for (const { modelType, providerKlass } of supportedModelTypes) {
      const cfProvider = await loadApiProvider(`cloudflare-ai:${modelType}:${modelName}`);
      const modelTypeForId: (typeof supportedModelTypes)[number]['modelType'] =
        modelType === 'embeddings' ? 'embedding' : modelType;

      expect(cfProvider.id()).toMatch(`cloudflare-ai:${modelTypeForId}:${modelName}`);
      expect(cfProvider).toBeInstanceOf(providerKlass);
    }
  });

  test('loadApiProvider with RawProviderConfig', async () => {
    const rawProviderConfig = {
      'openai:chat': {
        id: 'test',
        config: { foo: 'bar' },
      },
    };
    const provider = await loadApiProvider('openai:chat', {
      options: rawProviderConfig['openai:chat'],
    });
    expect(provider).toBeInstanceOf(OpenAiChatCompletionProvider);
  });

  test('loadApiProviders with ProviderFunction', async () => {
    const providerFunction: ProviderFunction = async (prompt: string) => {
      return {
        output: `Output for ${prompt}`,
        tokenUsage: { total: 10, prompt: 5, completion: 5 },
      };
    };
    const providers = await loadApiProviders(providerFunction);
    expect(providers).toHaveLength(1);
    expect(providers[0].id()).toBe('custom-function');
    const response = await providers[0].callApi('Test prompt');
    expect(response.output).toBe('Output for Test prompt');
    expect(response.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
  });

  test('loadApiProviders with RawProviderConfig[]', async () => {
    const rawProviderConfigs: ProviderOptionsMap[] = [
      {
        'openai:chat:abc123': {
          config: { foo: 'bar' },
        },
      },
      {
        'openai:completion:def456': {
          config: { foo: 'bar' },
        },
      },
      {
        'anthropic:completion:ghi789': {
          config: { foo: 'bar' },
        },
      },
    ];
    const providers = await loadApiProviders(rawProviderConfigs);
    expect(providers).toHaveLength(3);
    expect(providers[0]).toBeInstanceOf(OpenAiChatCompletionProvider);
    expect(providers[1]).toBeInstanceOf(OpenAiCompletionProvider);
    expect(providers[2]).toBeInstanceOf(AnthropicCompletionProvider);
  });

  test('loadApiProvider sets provider.delay', async () => {
    const providerOptions = {
      id: 'test-delay',
      config: {},
      delay: 500,
    };
    const provider = await loadApiProvider('echo', { options: providerOptions });
    expect(provider.delay).toBe(500);
  });
});
