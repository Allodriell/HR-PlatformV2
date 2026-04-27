export type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export type ApiErrorPayload = {
  code?: string;
  details?: unknown;
  message?: string;
};

export type Candidate = {
  id: string;
  fullName: string;
  headline?: string;
  location?: string;
  score?: number;
};

export type CandidateSearchRequest = {
  query: string;
  limit?: number;
};

export type CandidateSearchResponse = {
  items: Candidate[];
  total: number;
};

export type AssistantPromptRequest = {
  message: string;
  mode?: "chat" | "agent";
};

export type AssistantPromptResponse = {
  answer: string;
  conversationId?: string;
};

export type SpeechToTextResponse = {
  text: string;
};
