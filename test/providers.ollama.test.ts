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

describe('Ollama', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('OllamaCompletionProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn()
        .mockResolvedValue(`{"model":"llama2:13b","created_at":"2023-08-08T21:50:34.898068Z","response":"Gre","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:34.929199Z","response":"at","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:34.959989Z","response":" question","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:34.992117Z","response":"!","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:35.023658Z","response":" The","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:35.0551Z","response":" sky","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:35.086103Z","response":" appears","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:35.117166Z","response":" blue","done":false}
{"model":"llama2:13b","created_at":"2023-08-08T21:50:41.695299Z","done":true,"context":[1,29871,1,13,9314],"total_duration":10411943458,"load_duration":458333,"sample_count":217,"sample_duration":154566000,"prompt_eval_count":11,"prompt_eval_duration":3334582000,"eval_count":216,"eval_duration":6905134000}`),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OllamaCompletionProvider('llama');
    const result = await provider.callApi('Test prompt');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Great question! The sky appears blue');
  });

  test('OllamaChatProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn()
        .mockResolvedValue(`{"model":"orca-mini","created_at":"2023-12-16T01:46:19.263682972Z","message":{"role":"assistant","content":" Because","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.275143974Z","message":{"role":"assistant","content":" of","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.288137727Z","message":{"role":"assistant","content":" Ray","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.301139709Z","message":{"role":"assistant","content":"leigh","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.311364699Z","message":{"role":"assistant","content":" scattering","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.324309782Z","message":{"role":"assistant","content":".","images":null},"done":false}
{"model":"orca-mini","created_at":"2023-12-16T01:46:19.337165395Z","done":true,"total_duration":1486443841,"load_duration":1280794143,"prompt_eval_count":35,"prompt_eval_duration":142384000,"eval_count":6,"eval_duration":61912000}`),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new OllamaChatProvider('llama');
    const result = await provider.callApi('Test prompt');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe(' Because of Rayleigh scattering.');
  });
});
