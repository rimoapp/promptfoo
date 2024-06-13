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

describe('llama', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('LlamaProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          content: 'Test output',
        }),
      ),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new LlamaProvider('llama.cpp');
    const result = await provider.callApi('Test prompt');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
  });
});
