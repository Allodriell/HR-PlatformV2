import { httpClient } from "../http-client";
import type {
  AssistantPromptRequest,
  AssistantPromptResponse,
  SpeechToTextResponse,
} from "../types";

export const assistantApi = {
  sendPrompt(payload: AssistantPromptRequest) {
    return httpClient.post<AssistantPromptResponse>("/assistant/messages", {
      body: payload,
    });
  },

  transcribeAudio(file: Blob) {
    const formData = new FormData();
    formData.append("file", file, "recording.webm");

    return httpClient.post<SpeechToTextResponse>("/assistant/stt", {
      body: formData,
    });
  },
};
