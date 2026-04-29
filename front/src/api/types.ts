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
  tags?: string[];
};

export type CandidateCreateRequest = {
  email?: string;
  full_name: string;
  phone?: string;
  raw_resume_text: string;
  role?: string;
};

export type CandidateCreateResponse = {
  candidate: {
    candidate_id: number;
    chunks_count: number;
    email: string;
    full_name: string;
    phone: string;
    resume_id: number;
    role: string;
    tags?: string[];
  };
  message: string;
};

export type CandidateDetailResponse = {
  candidate: {
    candidate_id: number;
    email?: string;
    full_name: string;
    phone?: string;
    role?: string;
    tags?: string[];
  };
  resume_chunks: string[];
  resume_text: string;
};

export type CandidateQuestionRequest = {
  history?: Array<{
    content: string;
    role: "user" | "assistant";
  }>;
  question: string;
};

export type CandidateQuestionResponse = {
  answer: string;
  candidate: unknown;
  evidence_quote?: string;
  history?: Array<{
    content: string;
    role: string;
  }>;
};

export type CandidateSearchRequest = {
  query: string;
  limit?: number;
};

export type CandidateSearchResponse = {
  items: Candidate[];
  meta?: {
    normalized_query?: string;
  };
  total: number;
};

export type AssistantPromptRequest = {
  current_prompt?: string;
  decide_only?: boolean;
  message: string;
  mode?: "chat" | "agent";
};

export type AssistantPromptResponse = {
  action?: "needs_clarification" | "search_results";
  answer: string;
  chips?: string[];
  conversationId?: string;
  normalized_query?: string;
  search?: {
    candidates: Array<{
      candidate_id: number;
      email?: string;
      full_name: string;
      role?: string;
      tags?: string[];
      total_score?: number;
    }>;
    normalized_query: string;
  } | null;
};

export type SpeechToTextResponse = {
  text: string;
};
