import fetch from 'node-fetch';
import { clearCache, disableCache, enableCache } from '../src/cache';
import { loadApiProvider } from '../src/providers';
import {
  AzureOpenAiChatCompletionProvider,
  AzureOpenAiCompletionProvider,
} from '../src/providers/azureopenai';
import { AwsBedrockCompletionProvider } from '../src/providers/bedrock';
import { OpenAiCompletionProvider, OpenAiChatCompletionProvider } from '../src/providers/openai';

jest.mock('glob', () => ({
  globSync: jest.fn(),
}));

jest.mock('node-fetch', () => jest.fn());
jest.mock('proxy-agent', () => ({
  ProxyAgent: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../src/database');

describe('OpenAI Providers', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  describe('OpenAiCompletionProvider callApi', () => {
    it('should return output for default behavior', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ text: 'Test output' }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiCompletionProvider('text-davinci-003');
      const result = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
    });

    it('should return cached output with caching enabled', async () => {
      enableCache();
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ text: 'Test output' }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiCompletionProvider('text-davinci-003');
      const result = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

      (jest.mocked(fetch)).mockClear();
      const cachedResult = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(0);
      expect(cachedResult).toMatchObject(result);
    });

    it('should return fresh output with caching disabled', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ text: 'Test output' }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiCompletionProvider('text-davinci-003');
      const result = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

      (jest.mocked(fetch)).mockClear();
      disableCache();

      const freshResult = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(freshResult.output).toBe('Test output');
      expect(freshResult.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
      enableCache();
    });

    it('should handle missing apiKey', async () => {
      const provider = new OpenAiCompletionProvider('text-davinci-003', {
        config: { apiKey: undefined },
      });

      await expect(provider.callApi('Test prompt')).resolves.toEqual({
        error:
          'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      });
    });

    it('should handle API call error', async () => {
      const provider = new OpenAiCompletionProvider('text-davinci-003');
      provider.callApi = jest.fn().mockRejectedValue(new Error('API call failed'));

      const result = await provider.callApi('Test prompt');
      expect(result).toMatchObject({
        error: 'API call error: Error: API call failed',
      });
    });
  });

  describe('OpenAiChatCompletionProvider callApi', () => {
    it('should return output for default behavior', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ message: { content: 'Test output' } }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
        ok: true,
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
      const result = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
    });

    it('should return cached output with caching enabled', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ message: { content: 'Test output 2' } }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
        ok: true,
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
      const result = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt 2' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output 2');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

      const cachedResult = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt 2' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(cachedResult.output).toBe('Test output 2');
      expect(cachedResult.tokenUsage).toEqual({ total: 10, cached: 10 });
    });

    it('should return fresh output with caching disabled', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ message: { content: 'Test output' } }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
        ok: true,
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
      const result = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

      disableCache();

      const freshResult = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(2);
      expect(freshResult.output).toBe('Test output');
      expect(freshResult.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
      enableCache();
    });

    it('should handle missing apiKey', async () => {
      const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo', {
        config: { apiKey: undefined },
      });

      await expect(provider.callApi('Test prompt')).resolves.toEqual({
        error:
          'OpenAI API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the provider config.',
      });
    });

    it('should handle API call error', async () => {
      const provider = new OpenAiChatCompletionProvider('gpt-3.5-turbo');
      provider.callApi = jest.fn().mockRejectedValue(new Error('API call failed'));

      const result = await provider.callApi('Test prompt');
      expect(result).toMatchObject({
        error: 'API call error: Error: API call failed',
      });
    });
  });

  describe('AzureOpenAiCompletionProvider callApi', () => {
    it('should return output for default behavior', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ text: 'Test output' }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new AzureOpenAiCompletionProvider('text-davinci-003');
      const result = await provider.callApi('Test prompt');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
    });
  });

  describe('AzureOpenAiChatCompletionProvider callApi', () => {
    it('should return output for default behavior', async () => {
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ message: { content: 'Test output' } }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new AzureOpenAiChatCompletionProvider('gpt-3.5-turbo');
      const result = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
    });

    it('should return output with dataSources', async () => {
      const dataSources = [
        {
          type: 'AzureCognitiveSearch',
          endpoint: 'https://search.windows.net',
          indexName: 'search-test',
          semanticConfiguration: 'default',
          queryType: 'vectorSimpleHybrid',
        },
      ];
      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [
              { message: { role: 'system', content: 'System prompt' } },
              { message: { role: 'user', content: 'Test prompt' } },
              { message: { role: 'assistant', content: 'Test response' } },
            ],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new AzureOpenAiChatCompletionProvider('gpt-3.5-turbo', {
        config: { dataSources },
      });
      const result = await provider.callApi(
        JSON.stringify([
          { role: 'system', content: 'System prompt' },
          { role: 'user', content: 'Test prompt' },
        ]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test response');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });
    });

    it('should return fresh output with cache disabled', async () => {
      disableCache();

      const mockResponse = {
        text: jest.fn().mockResolvedValue(
          JSON.stringify({
            choices: [{ message: { content: 'Test output' } }],
            usage: { total_tokens: 10, prompt_tokens: 5, completion_tokens: 5 },
          }),
        ),
      };
      (jest.mocked(fetch)).mockResolvedValue(mockResponse);

      const provider = new AzureOpenAiChatCompletionProvider('gpt-3.5-turbo');
      const result = await provider.callApi(
        JSON.stringify([{ role: 'user', content: 'Test prompt' }]),
      );

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result.output).toBe('Test output');
      expect(result.tokenUsage).toEqual({ total: 10, prompt: 5, completion: 5 });

      enableCache();
    });
  });

  // NOTE: test suite fails with: ReferenceError: Cannot access 'AnthropicCompletionProvider' before initialization
  // if this is removed. The test can even be skipped. This is likely due to a circular dependency.
  it('loadApiProvider with bedrock:completion', async () => {
    await expect(
      loadApiProvider('bedrock:completion:anthropic.claude-v2:1'),
    ).resolves.toBeInstanceOf(AwsBedrockCompletionProvider);
  });
});
