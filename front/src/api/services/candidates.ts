import { httpClient } from "../http-client";
import type {
  CandidateCreateRequest,
  CandidateCreateResponse,
  CandidateDetailResponse,
  CandidateQuestionRequest,
  CandidateQuestionResponse,
  CandidateSearchRequest,
  CandidateSearchResponse,
} from "../types";

export const candidatesApi = {
  create(payload: CandidateCreateRequest) {
    return httpClient.post<CandidateCreateResponse>("/candidates", {
      body: payload,
    });
  },

  get(candidateId: string | number) {
    return httpClient.get<CandidateDetailResponse>(`/candidates/${candidateId}`);
  },

  askQuestion(candidateId: string | number, payload: CandidateQuestionRequest) {
    return httpClient.post<CandidateQuestionResponse>(`/candidates/${candidateId}/qa`, {
      body: payload,
    });
  },

  search(params: CandidateSearchRequest) {
    return httpClient.get<CandidateSearchResponse>("/candidates/search", {
      query: {
        limit: params.limit,
        query: params.query,
      },
    });
  },
};
