import { clearCache, disableCache, enableCache } from '../src/cache';
import { AnthropicCompletionProvider } from '../src/providers/anthropic';

describe('AnthropicCompletionProvider', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('callApi', async () => {
    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });
    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({});
  });

  test('callApi with caching', async () => {
    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });
    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({});

    (provider.anthropic.completions.create as jest.Mock).mockClear();
    const result2 = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(0);
    expect(result2.output).toBe('Test output');
    expect(result2.tokenUsage).toEqual({});
  });

  test('callApi with caching disabled', async () => {
    const provider = new AnthropicCompletionProvider('claude-1');
    provider.anthropic.completions.create = jest.fn().mockResolvedValue({
      completion: 'Test output',
    });
    const result = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
    expect(result.tokenUsage).toEqual({});

    (provider.anthropic.completions.create as jest.Mock).mockClear();

    disableCache();

    const result2 = await provider.callApi('Test prompt');

    expect(provider.anthropic.completions.create).toHaveBeenCalledTimes(1);
    expect(result2.output).toBe('Test output');
    expect(result2.tokenUsage).toEqual({});
  });
});
