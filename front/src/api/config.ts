export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000/api";

export const API_TIMEOUT_MS = 15_000;
