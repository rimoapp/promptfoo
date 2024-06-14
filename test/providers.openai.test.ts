import OpenAI from 'openai';
import logger from '../src/logger';
import { fetchWithCache, getCache, isCacheEnabled } from '../src/cache';
import { renderVarsInObject } from '../src/util';
import { REQUEST_TIMEOUT_MS, parseChatPrompt, toTitleCase } from '../src/providers/shared';
import { OpenAiFunction, OpenAiTool } from '../src/providers/openaiUtil';
import { safeJsonStringify } from '../src/util';
import {
  OpenAiGenericProvider,
  OpenAiEmbeddingProvider,
  OpenAiCompletionProvider,
  OpenAiChatCompletionProvider,
  OpenAiImageProvider,
  OpenAiModerationProvider,
  DefaultEmbeddingProvider,
  DefaultGradingProvider,
  DefaultGradingJsonProvider,
  DefaultSuggestionsProvider,
  DefaultModerationProvider,
} from '../src/providers/openai';
import type { Cache } from 'cache-manager';

import type {
  ApiModerationProvider,
  ApiProvider,
  CallApiContextParams,
  CallApiOptionsParams,
  EnvOverrides,
  ModerationFlag,
  ProviderEmbeddingResponse,
  ProviderModerationResponse,
  ProviderResponse,
  TokenUsage,
} from '../src/types';

interface OpenAiSharedOptions {
  apiKey?: string;
  apiKeyEnvar?: string;
  apiHost?: string;
  apiBaseUrl?: string;
  organization?: string;
  cost?: number;
  headers?: { [key: string]: string };
}

type OpenAiCompletionOptions = OpenAiSharedOptions & {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  best_of?: number;
  functions?: OpenAiFunction[];
  function_call?: 'none' | 'auto' | { name: string };
  tools?: OpenAiTool[];
  tool_choice?: 'none' | 'auto' | 'required' | { type: 'function'; function?: { name: string } };
  response_format?: { type: 'json_object' };
  stop?: string[];
  seed?: number;
  passthrough?: object;

  /**
   * If set, automatically call these functions when the assistant activates
   * these function tools.
   */
  functionToolCallbacks?: Record<
    OpenAI.FunctionDefinition['name'],
    (arg: string) => Promise<string>
  >;
};

function failApiCall(err: any) {
  if (err instanceof OpenAI.APIError) {
    return {
      error: `API error: ${err.type} ${err.message}`,
    };
  }
  return {
    error: `API error: ${String(err)}`,
  };
}

function getTokenUsage(data: any, cached: boolean): Partial<TokenUsage> {
  if (data.usage) {
    if (cached) {
      return { cached: data.usage.total_tokens, total: data.usage.total_tokens };
    } else {
      return {
        total: data.usage.total_tokens,
        prompt: data.usage.prompt_tokens || 0,
        completion: data.usage.completion_tokens || 0,
      };
    }
  }
  return {};
}

// Tests
describe('OpenAiGenericProvider', () => {
  describe('constructor', () => {
    it('should initialize with the correct values', () => {
      const provider = new OpenAiGenericProvider('testModel', { config: { apiKey: 'testKey' }, id: 'testId' });
      expect(provider.modelName).toBe('testModel');
      expect(provider.config.apiKey).toBe('testKey');
      expect(provider.id()).toBe('testId');
    });

    it('should initialize without config and id', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.modelName).toBe('testModel');
      expect(provider.config).toMatchObject({});
    });
  });

  describe('id', () => {
    it('should return the correct id based on the configuration', () => {
      const provider = new OpenAiGenericProvider('testModel', { config: { apiHost: 'api.test.com' } });
      expect(provider.id()).toBe('testModel');
    });

    it('should return default id if no apiHost or apiBaseUrl is provided', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.id()).toBe('openai:testModel');
    });
  });

  describe('getOrganization', () => {
    it('should return the correct organization based on the configuration and environment variables', () => {
      process.env.OPENAI_ORGANIZATION = 'envOrg';
      const provider = new OpenAiGenericProvider('testModel', { config: { organization: 'configOrg' } });
      expect(provider.getOrganization()).toBe('configOrg');
    });

    it('should return the environment variable organization if no config is provided', () => {
      process.env.OPENAI_ORGANIZATION = 'envOrg';
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getOrganization()).toBe('envOrg');
    });

    it('should return undefined if no organization is provided', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getOrganization()).toBeUndefined();
    });
  });

  describe('getApiUrlDefault', () => {
    it('should return the default API URL', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getApiUrlDefault()).toBe('https://api.openai.com/v1');
    });
  });

  describe('getApiUrl', () => {
    it('should return the correct API URL based on the configuration and environment variables', () => {
      process.env.OPENAI_API_HOST = 'env.api.host';
      const provider = new OpenAiGenericProvider('testModel', { config: { apiHost: 'config.api.host' } });
      expect(provider.getApiUrl()).toBe('https://config.api.host/v1');
    });

    it('should return the default API URL if no config or environment variable is provided', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getApiUrl()).toBe('https://api.openai.com/v1');
    });
  });

  describe('getApiKey', () => {
    it('should return the correct API key based on the configuration and environment variables', () => {
      process.env.OPENAI_API_KEY = 'envApiKey';
      const provider = new OpenAiGenericProvider('testModel', { config: { apiKey: 'configApiKey' } });
      expect(provider.getApiKey()).toBe('configApiKey');
    });

    it('should return the environment variable API key if no config is provided', () => {
      process.env.OPENAI_API_KEY = 'envApiKey';
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getApiKey()).toBe('envApiKey');
    });

    it('should return undefined if no API key is provided', () => {
      const provider = new OpenAiGenericProvider('testModel');
      expect(provider.getApiKey()).toBeUndefined();
    });
  });

  describe('callApi', () => {
    it('should throw an error indicating the method is not implemented', async () => {
      const provider = new OpenAiGenericProvider('testModel');
      await expect(provider.callApi('testPrompt')).rejects.toThrow('Not implemented');
    });
  });
});

describe('OpenAiEmbeddingProvider', () => {
  describe('callEmbeddingApi', () => {
    it('should throw an error if API key is not set', async () => {
      const provider = new OpenAiEmbeddingProvider('testModel');
      await expect(provider.callEmbeddingApi('testText')).rejects.toThrow('OpenAI API key must be set for similarity comparison');
    });

    it('should call the OpenAI embeddings API and return the embedding', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { data: [{ embedding: [1, 2, 3] }], usage: { total_tokens: 5 } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiEmbeddingProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callEmbeddingApi('testText');
      expect(response).toMatchObject({
        embedding: [1, 2, 3],
        tokenUsage: { total: 5, prompt: 0, completion: 0 }
      });
    });

    it('should handle API call errors', async () => {
      const mockFetchWithCache = jest.fn().mockRejectedValue(new Error('API error'));
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiEmbeddingProvider('testModel', { config: { apiKey: 'testKey' } });
      await expect(provider.callEmbeddingApi('testText')).rejects.toThrow('API error');
    });

    it('should handle missing embedding in the API response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { data: [] },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiEmbeddingProvider('testModel', { config: { apiKey: 'testKey' } });
      await expect(provider.callEmbeddingApi('testText')).rejects.toThrow('No embedding found in OpenAI embeddings API response');
    });
  });
});

describe('OpenAiCompletionProvider', () => {
  describe('constructor', () => {
    it('should initialize with the correct values and warn for unknown model', () => {
      const mockWarn = jest.spyOn(logger, 'warn').mockImplementation(() => {});
      const provider = new OpenAiCompletionProvider('unknownModel', { config: { apiKey: 'testKey' } });
      expect(provider.modelName).toBe('unknownModel');
      expect(provider.config.apiKey).toBe('testKey');
      expect(mockWarn).toHaveBeenCalledWith('FYI: Using unknown OpenAI completion model: unknownModel');
      mockWarn.mockRestore();
    });
  });

  describe('callApi', () => {
    it('should throw an error if API key is not set', async () => {
      const provider = new OpenAiCompletionProvider('testModel');
      await expect(provider.callApi('testPrompt')).rejects.toThrow('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
    });

    it('should call the OpenAI completions API and return the response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { choices: [{ text: 'test response' }], usage: { total_tokens: 5 } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        output: 'test response',
        tokenUsage: { total: 5, prompt: 0, completion: 0 },
        cached: false
      });
    });

    it('should handle API call errors', async () => {
      const mockFetchWithCache = jest.fn().mockRejectedValue(new Error('API error'));
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'API call error: Error: API error'
      });
    });

    it('should handle errors in the API response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { error: { message: 'test error', type: 'test type' } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'API error: test type: test error'
      });
    });
  });
});

describe('OpenAiChatCompletionProvider', () => {
  describe('constructor', () => {
    it('should initialize with the correct values and warn for unknown model', () => {
      const mockDebug = jest.spyOn(logger, 'debug').mockImplementation(() => {});
      const provider = new OpenAiChatCompletionProvider('unknownModel', { config: { apiKey: 'testKey' } });
      expect(provider.modelName).toBe('unknownModel');
      expect(provider.config.apiKey).toBe('testKey');
      expect(mockDebug).toHaveBeenCalledWith('Using unknown OpenAI chat model: unknownModel');
      mockDebug.mockRestore();
    });
  });

  describe('callApi', () => {
    it('should throw an error if API key is not set', async () => {
      const provider = new OpenAiChatCompletionProvider('testModel');
      await expect(provider.callApi('testPrompt')).rejects.toThrow('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
    });

    it('should call the OpenAI chat completions API and return the response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { choices: [{ message: { content: 'test response' } }], usage: { total_tokens: 5 } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiChatCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        output: 'test response',
        tokenUsage: { total: 5, prompt: 0, completion: 0 },
        cached: false
      });
    });

    it('should handle API call errors', async () => {
      const mockFetchWithCache = jest.fn().mockRejectedValue(new Error('API error'));
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiChatCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'API call error: Error: API error'
      });
    });

    it('should handle errors in the API response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { error: { message: 'test error', type: 'test type' } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiChatCompletionProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'API error: test type: test error'
      });
    });
  });
});

describe('OpenAiImageProvider', () => {
  describe('callApi', () => {
    it('should throw an error if API key is not set', async () => {
      const provider = new OpenAiImageProvider('testModel');
      await expect(provider.callApi('testPrompt')).rejects.toThrow('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
    });

    it('should call the OpenAI image generation API and return the image URL', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { data: [{ url: 'http://test.url/image.png' }] },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiImageProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        output: '![testPrompt](http://test.url/image.png)',
        cached: false
      });
    });

    it('should handle API call errors', async () => {
      const mockFetchWithCache = jest.fn().mockRejectedValue(new Error('API error'));
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiImageProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'API call error: Error: API error'
      });
    });

    it('should handle missing image URL in the API response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { data: [{}] },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiImageProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callApi('testPrompt');
      expect(response).toMatchObject({
        error: 'No image URL found in response: {"data":[{}]}'
      });
    });
  });
});

describe('OpenAiModerationProvider', () => {
  describe('callModerationApi', () => {
    it('should throw an error if API key is not set', async () => {
      const provider = new OpenAiModerationProvider('testModel');
      await expect(provider.callModerationApi('testPrompt', 'testResponse')).rejects.toThrow('OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.');
    });

    it('should call the OpenAI moderation API and return the response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { results: [{ flagged: true, categories: { hate: true }, category_scores: { hate: 0.9 } }] },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiModerationProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callModerationApi('testPrompt', 'testResponse');
      expect(response).toMatchObject({
        flags: [{ code: 'hate', description: 'hate', confidence: 0.9 }]
      });
    });

    it('should handle API call errors', async () => {
      const mockFetchWithCache = jest.fn().mockRejectedValue(new Error('API error'));
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiModerationProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callModerationApi('testPrompt', 'testResponse');
      expect(response).toMatchObject({
        error: 'API call error: Error: API error'
      });
    });

    it('should handle errors in the API response', async () => {
      const mockFetchWithCache = jest.fn().mockResolvedValue({
        data: { error: { message: 'test error', type: 'test type' } },
        cached: false
      });
      jest.mock('../cache', () => ({ fetchWithCache: mockFetchWithCache }));

      const provider = new OpenAiModerationProvider('testModel', { config: { apiKey: 'testKey' } });
      const response = await provider.callModerationApi('testPrompt', 'testResponse');
      expect(response).toMatchObject({
        error: 'API error: test type: test error'
      });
    });
  });
});

describe('Utility Functions', () => {
  describe('failApiCall', () => {
    it('should return the correct error message for APIError', () => {
      const error = new OpenAI.APIError('test error', 'test type');
      const response = failApiCall(error);
      expect(response).toMatchObject({ error: 'API error: test type test error' });
    });

    it('should return the correct error message for other errors', () => {
      const error = new Error('test error');
      const response = failApiCall(error);
      expect(response).toMatchObject({ error: 'API error: Error: test error' });
    });
  });

  describe('getTokenUsage', () => {
    it('should return the correct token usage for cached data', () => {
      const data = { usage: { total_tokens: 5 } };
      const response = getTokenUsage(data, true);
      expect(response).toMatchObject({ cached: 5, total: 5 });
    });

    it('should return the correct token usage for non-cached data', () => {
      const data = { usage: { total_tokens: 5, prompt_tokens: 3, completion_tokens: 2 } };
      const response = getTokenUsage(data, false);
      expect(response).toMatchObject({ total: 5, prompt: 3, completion: 2 });
    });

    it('should return an empty object if no usage data is present', () => {
      const data = {};
      const response = getTokenUsage(data, false);
      expect(response).toMatchObject({});
    });
  });
});
