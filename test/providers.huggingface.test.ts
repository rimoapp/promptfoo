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

describe('Huggingface', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  describe.each([
    ['Array format', [{ generated_text: 'Test output' }]], // Array format
    ['Object format', { generated_text: 'Test output' }], // Object format
  ])('HuggingfaceTextGenerationProvider callApi with %s', (format, mockedData) => {
    test('returns expected output', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(JSON.stringify(mockedData)),
      };
      (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

      const provider = new HuggingfaceTextGenerationProvider('gpt2');
      const result = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
    });
  });

  test('HuggingfaceFeatureExtractionProvider callEmbeddingApi', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(JSON.stringify([0.1, 0.2, 0.3, 0.4, 0.5])),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new HuggingfaceFeatureExtractionProvider('distilbert-base-uncased');
    const result = await provider.callEmbeddingApi('Test text');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.embedding).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
  });

  test('HuggingfaceTextClassificationProvider callClassificationApi', async () => {
    const mockClassification = [
      [
        {
          label: 'nothate',
          score: 0.9,
        },
        {
          label: 'hate',
          score: 0.1,
        },
      ],
    ];
    const mockResponse = {
      text: jest.fn().mockResolvedValue(JSON.stringify(mockClassification)),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new HuggingfaceTextClassificationProvider('foo');
    const result = await provider.callClassificationApi('Test text');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.classification).toEqual({
      nothate: 0.9,
      hate: 0.1,
    });
  });
});
