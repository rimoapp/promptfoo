import fetch from 'node-fetch';
import { WebhookProvider } from '../src/providers/webhook';
import { clearCache } from '../src/cache';

jest.mock('node-fetch', () => jest.fn());

describe('WebhookProvider', () => {
  afterEach(async () => {
    jest.clearAllMocks();
    await clearCache();
  });

  test('WebhookProvider callApi', async () => {
    const mockResponse = {
      text: jest.fn().mockResolvedValue(
        JSON.stringify({
          output: 'Test output',
        }),
      ),
    };
    (fetch as unknown as jest.Mock).mockResolvedValue(mockResponse);

    const provider = new WebhookProvider('http://example.com/webhook');
    const result = await provider.callApi('Test prompt');

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(result.output).toBe('Test output');
  });
});
