import * as fs from 'fs';
import fetch from 'node-fetch';
import child_process from 'child_process';
import Stream from 'stream';

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
import { ScriptCompletionProvider } from '../src/providers/scriptCompletion';
import {
  CloudflareAiChatCompletionProvider,
  CloudflareAiCompletionProvider,
  CloudflareAiEmbeddingProvider,
  type ICloudflareProviderBaseConfig,
  type ICloudflareTextGenerationResponse,
  type ICloudflareEmbeddingResponse,
  type ICloudflareProviderConfig,
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

describe('CloudflareAi', () => {
  beforeAll(() => {
    enableCache();
  });

  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  const fetchMock = fetch as unknown as jest.Mock;
  const cloudflareMinimumConfig: Required<
    Pick<ICloudflareProviderBaseConfig, 'accountId' | 'apiKey'>
  > = {
    accountId: 'testAccountId',
    apiKey: 'testApiKey',
  };

  const testModelName = '@cf/meta/llama-2-7b-chat-fp16';
  // Token usage is not implemented for cloudflare so this is the default that
  // is returned
  const tokenUsageDefaultResponse = {
    total: undefined,
    prompt: undefined,
    completion: undefined,
  };

  describe('CloudflareAiCompletionProvider', () => {
    test('callApi with caching enabled', async () => {
      const PROMPT = 'Test prompt for caching';
      const provider = new CloudflareAiCompletionProvider(testModelName, {
        config: cloudflareMinimumConfig,
      });

      const responsePayload: ICloudflareTextGenerationResponse = {
        success: true,
        errors: [],
        messages: [],
        result: {
          response: 'Test text output',
        },
      };
      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
        ok: true,
      };

      fetchMock.mockResolvedValue(mockResponse);
      const result = await provider.callApi(PROMPT);

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe(responsePayload.result.response);
      expect(result.tokenUsage).toEqual(tokenUsageDefaultResponse);

      const resultFromCache = await provider.callApi(PROMPT);

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(resultFromCache.output).toBe(responsePayload.result.response);
      expect(resultFromCache.tokenUsage).toEqual(tokenUsageDefaultResponse);
    });

    test('callApi with caching disabled', async () => {
      const PROMPT = 'test prompt without caching';
      try {
        disableCache();
        const provider = new CloudflareAiCompletionProvider(testModelName, {
          config: cloudflareMinimumConfig,
        });

        const responsePayload: ICloudflareTextGenerationResponse = {
          success: true,
          errors: [],
          messages: [],
          result: {
            response: 'Test text output',
          },
        };
        const mockResponse = {
          text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
          ok: true,
        };

        fetchMock.mockResolvedValue(mockResponse);
        const result = await provider.callApi(PROMPT);

        expect(fetch).toHaveBeenCalledTimes(1);
        expect(result.output).toBe(responsePayload.result.response);
        expect(result.tokenUsage).toEqual(tokenUsageDefaultResponse);

        const resultFromCache = await provider.callApi(PROMPT);
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(resultFromCache.output).toBe(responsePayload.result.response);
        expect(resultFromCache.tokenUsage).toEqual(tokenUsageDefaultResponse);
      } finally {
        enableCache();
      }
    });

    test('callApi handles cloudflare error properly', async () => {
      const PROMPT = 'Test prompt for caching';
      const provider = new CloudflareAiCompletionProvider(testModelName, {
        config: cloudflareMinimumConfig,
      });

      const responsePayload: ICloudflareTextGenerationResponse = {
        success: false,
        errors: ['Some error occurred'],
        messages: [],
      };
      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
        ok: true,
      };

      fetchMock.mockResolvedValue(mockResponse);
      const result = await provider.callApi(PROMPT);

      expect(result.error).toContain(JSON.stringify(responsePayload.errors));
    });

    test('Can be invoked with custom configuration', async () => {
      const cloudflareChatConfig: ICloudflareProviderConfig = {
        accountId: 'MADE_UP_ACCOUNT_ID',
        apiKey: 'MADE_UP_API_KEY',
        frequency_penalty: 10,
      };
      const rawProviderConfigs: ProviderOptionsMap[] = [
        {
          [`cloudflare-ai:completion:${testModelName}`]: {
            config: cloudflareChatConfig,
          },
        },
      ];

      const providers = await loadApiProviders(rawProviderConfigs);
      expect(providers).toHaveLength(1);
      expect(providers[0]).toBeInstanceOf(CloudflareAiCompletionProvider);

      const cfProvider = providers[0] as CloudflareAiCompletionProvider;
      expect(cfProvider.config).toEqual(cloudflareChatConfig);

      const PROMPT = 'Test prompt for custom configuration';

      const responsePayload: ICloudflareTextGenerationResponse = {
        success: true,
        errors: [],
        messages: [],
        result: {
          response: 'Test text output',
        },
      };
      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
        ok: true,
      };

      fetchMock.mockResolvedValue(mockResponse);
      await cfProvider.callApi(PROMPT);

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls.length).toBe(1);
      const [url, { body, headers, method }] = fetchMock.mock.calls[0];
      expect(url).toContain(cloudflareChatConfig.accountId);
      expect(method).toBe('POST');
      expect(headers['Authorization']).toContain(cloudflareChatConfig.apiKey);
      const hydratedBody = JSON.parse(body);
      expect(hydratedBody.prompt).toBe(PROMPT);

      const { accountId, apiKey, ...passThroughConfig } = cloudflareChatConfig;
      const { prompt, ...bodyWithoutPrompt } = hydratedBody;
      expect(bodyWithoutPrompt).toEqual(passThroughConfig);
    });
  });

  describe('CloudflareAiChatCompletionProvider', () => {
    test('Should handle chat provider', async () => {
      const provider = new CloudflareAiChatCompletionProvider(testModelName, {
        config: cloudflareMinimumConfig,
      });

      const responsePayload: ICloudflareTextGenerationResponse = {
        success: true,
        errors: [],
        messages: [],
        result: {
          response: 'Test text output',
        },
      };
      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
        ok: true,
      };

      fetchMock.mockResolvedValue(mockResponse);
      const result = await provider.callApi('Test chat prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe(responsePayload.result.response);
      expect(result.tokenUsage).toEqual(tokenUsageDefaultResponse);
    });
  });

  describe('CloudflareAiEmbeddingProvider', () => {
    test('Should return embeddings in the proper format', async () => {
      const provider = new CloudflareAiEmbeddingProvider(testModelName, {
        config: cloudflareMinimumConfig,
      });

      const responsePayload: ICloudflareEmbeddingResponse = {
        success: true,
        errors: [],
        messages: [],
        result: {
          shape: [1, 3],
          data: [[0.02055364102125168, -0.013749595731496811, 0.0024201320484280586]],
        },
      };

      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(responsePayload)),
        ok: true,
      };

      fetchMock.mockResolvedValue(mockResponse);
      const result = await provider.callEmbeddingApi('Create embeddings from this');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.embedding).toEqual(responsePayload.result.data[0]);
      expect(result.tokenUsage).toEqual(tokenUsageDefaultResponse);
    });
  });
});
