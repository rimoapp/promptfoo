import { clearCache, disableCache } from '../src/cache';
import { AnthropicCompletionProvider } from '../src/providers/anthropic';

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
});
