import { EnvOverrides } from '../types';
import {
  DefaultGradingJsonProvider as AnthropicGradingJsonProvider,
  DefaultGradingProvider as AnthropicGradeProvider,
  DefaultSuggestionsProvider as AnthropicSuggestionsProvider,
} from './anthropic';
import {
  DefaultEmbeddingProvider as OpenAiEmbeddingProvider,
  DefaultGradingJsonProvider as OpenAiGradingJsonProvider,
  DefaultGradingProvider as OpenAiGradingProvider,
  DefaultModerationProvider as OpenAiModerationProvider,
  DefaultSuggestionsProvider as OpenAiSuggestionsProvider,
} from './openai';

export function getDefaultProviders(env?: EnvOverrides) {
  const preferAnthropic =
    !process.env.OPENAI_API_KEY &&
    !env?.OPENAI_API_KEY &&
    (process.env.ANTHROPIC_API_KEY || env?.ANTHROPIC_API_KEY);

  if (preferAnthropic) {
    return {
      embeddingProvider: OpenAiEmbeddingProvider, // TODO(ian): AnthropicEmbeddingProvider
      gradingProvider: AnthropicGradeProvider,
      gradingJsonProvider: AnthropicGradingJsonProvider,
      suggestionsProvider: AnthropicSuggestionsProvider,
      moderationProvider: OpenAiModerationProvider,
    };
  }
  return {
    embeddingProvider: OpenAiEmbeddingProvider,
    gradingProvider: OpenAiGradingProvider,
    gradingJsonProvider: OpenAiGradingJsonProvider,
    suggestionsProvider: OpenAiSuggestionsProvider,
    moderationProvider: OpenAiModerationProvider,
  };
}
