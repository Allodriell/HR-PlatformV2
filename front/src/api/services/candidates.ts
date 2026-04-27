import { httpClient } from "../http-client";
import type { CandidateSearchRequest, CandidateSearchResponse } from "../types";

export const candidatesApi = {
  search(params: CandidateSearchRequest) {
    return httpClient.get<CandidateSearchResponse>("/candidates/search", {
      query: {
        limit: params.limit,
        query: params.query,
      },
    });
  },
};
