import { clearCache, disableCache, enableCache } from '../src/cache';
import { AnthropicCompletionProvider, AnthropicMessagesProvider } from '../src/providers/anthropic';
import Anthropic, { APIError } from '@anthropic-ai/sdk';

describe('AnthropicCompletionProvider', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('callApi', async () => {
    expect.assertions(2);

    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });

    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });
  });

  test('callApi with caching', async () => {
    expect.assertions(4);

    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });

    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });

    (provider.anthropic.completions.create as jest.Mock).mockClear();

    const result2 = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(0);
    expect(result2).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });
  });

  test('callApi with caching disabled', async () => {
    expect.assertions(4);

    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });

    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });

    (provider.anthropic.completions.create as jest.Mock).mockClear();

    disableCache();

    const result2 = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result2).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });
  });

  test('callApi without API key', async () => {
    expect.assertions(1);

    const provider = new AnthropicCompletionProvider('claude-1', { config: {} });
    provider.apiKey = undefined;

    await expect(provider.callApi('Test prompt')).rejects.toThrow(
      'Anthropic API key is not set. Set the ANTHROPIC_API_KEY environment variable or add `apiKey` to the provider config.',
    );
  });

  test('callApi with API error', async () => {
    expect.assertions(2);

    const provider = new AnthropicCompletionProvider('claude-1');
    const apiError = new APIError('API call failed', 500, {
      error: { message: 'Internal Error', type: 'server_error' },
    });
    provider.anthropic.completions.create = jest.fn().mockRejectedValue(apiError);

    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      error: 'API call error: Error: API call failed 500',
    });
  });

  test('callApi with invalid stop sequences', async () => {
    expect.assertions(1);

    process.env.ANTHROPIC_STOP = 'invalid-json';

    const provider = new AnthropicCompletionProvider('claude-1');

    await expect(provider.callApi('Test prompt')).rejects.toThrow(
      'ANTHROPIC_STOP is not a valid JSON string: SyntaxError: Unexpected token \'i\', "invalid-json" is not valid JSON',
    );

    delete process.env.ANTHROPIC_STOP;
  });

  test('callApi with valid stop sequences', async () => {
    expect.assertions(2);

    process.env.ANTHROPIC_STOP = '["STOP"]';

    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });

    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledWith(
      expect.objectContaining({
        stop_sequences: ['STOP'],
      }),
    );
    expect(result).toMatchObject({
      output: 'Test output',
      tokenUsage: {},
    });

    delete process.env.ANTHROPIC_STOP;
  });

  test('callApi with token usage calculation', async () => {
    expect.assertions(2);

    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
      usage: { input_tokens: 50, output_tokens: 100 },
    });

    const result = await provider.callApi('Test prompt');

    expect(result.tokenUsage).toMatchObject({
      total: 150,
      prompt: 50,
      completion: 100,
    });
    expect(result.output).toBe('Test output');
  });

  test('callApi with cost calculation', async () => {
    expect.assertions(1);

    const provider = new AnthropicCompletionProvider('claude-instant-1.2', {
      config: { cost: 0.001 },
    });
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
      usage: { input_tokens: 50, output_tokens: 100 },
    });

    const result = await provider.callApi('Test prompt');

    expect(result.cost).toBeCloseTo(0.15);
  });
});

describe('AnthropicMessagesProvider', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('callApi with message format', async () => {
    expect.assertions(2);

    const provider = new AnthropicMessagesProvider('claude-2.0');
    provider.anthropic.messages.create = jest.fn().mockResolvedValue({
      content: [{ text: 'Test message output' }],
      usage: { input_tokens: 50, output_tokens: 100 },
    });

    const result = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test message output',
      tokenUsage: {
        total: 150,
        prompt: 50,
        completion: 100,
      },
    });
  });

  test('callApi with caching', async () => {
    expect.assertions(4);

    const provider = new AnthropicMessagesProvider('claude-2.0');
    provider.anthropic.messages.create = jest.fn().mockResolvedValue({
      content: [{ text: 'Test message output' }],
      usage: { input_tokens: 50, output_tokens: 100 },
    });

    const result = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test message output',
      tokenUsage: {
        total: 150,
        prompt: 50,
        completion: 100,
      },
    });

    (provider.anthropic.messages.create as jest.Mock).mockClear();

    const result2 = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(0);
    expect(result2).toMatchObject({
      output: 'Test message output',
      tokenUsage: {
        total: 150,
        prompt: 50,
        completion: 100,
      },
    });
  });

  test('callApi with caching disabled', async () => {
    expect.assertions(4);

    const provider = new AnthropicMessagesProvider('claude-2.0');
    provider.anthropic.messages.create = jest.fn().mockResolvedValue({
      content: [{ text: 'Test message output' }],
      usage: { input_tokens: 50, output_tokens: 100 },
    });

    const result = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      output: 'Test message output',
      tokenUsage: {
        total: 150,
        prompt: 50,
        completion: 100,
      },
    });

    (provider.anthropic.messages.create as jest.Mock).mockClear();

    disableCache();

    const result2 = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(1);
    expect(result2).toMatchObject({
      output: 'Test message output',
      tokenUsage: {
        total: 150,
        prompt: 50,
        completion: 100,
      },
    });
  });

  test('callApi without API key', async () => {
    expect.assertions(1);

    const provider = new AnthropicMessagesProvider('claude-2.0', { config: {} });
    provider.apiKey = undefined;

    await expect(provider.callApi('Test message prompt')).rejects.toThrow(
      'Anthropic API key is not set. Set the ANTHROPIC_API_KEY environment variable or add `apiKey` to the provider config.',
    );
  });

  test('callApi with API error', async () => {
    expect.assertions(2);

    const provider = new AnthropicMessagesProvider('claude-2.0');
    const apiError = new APIError('API call failed', 500, {
      error: { message: 'Internal Error', type: 'server_error' },
    });
    provider.anthropic.messages.create = jest.fn().mockRejectedValue(apiError);

    const result = await provider.callApi('Test message prompt');

    expect(provider.anthropic.messages.create).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      error: 'API call error: Error: API call failed 500',
    });
  });
});
