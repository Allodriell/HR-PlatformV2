import { API_BASE_URL, API_TIMEOUT_MS } from "./config";
import type { ApiErrorPayload, HttpMethod } from "./types";

type Primitive = string | number | boolean;

type QueryValue = Primitive | null | undefined;

type RequestOptions = {
  body?: BodyInit | object | null;
  headers?: HeadersInit;
  method?: HttpMethod;
  query?: Record<string, QueryValue>;
  signal?: AbortSignal;
  timeoutMs?: number;
};

export class ApiError extends Error {
  readonly details?: unknown;
  readonly status: number;

  constructor(status: number, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function buildUrl(path: string, query?: Record<string, QueryValue>) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = new URL(`${API_BASE_URL}${normalizedPath}`);

  if (!query) {
    return url.toString();
  }

  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }

    url.searchParams.set(key, String(value));
  }

  return url.toString();
}

function isBodyInit(value: BodyInit | object | null | undefined): value is BodyInit {
  return (
    typeof value === "string" ||
    value instanceof FormData ||
    value instanceof URLSearchParams ||
    value instanceof Blob ||
    value instanceof ArrayBuffer ||
    ArrayBuffer.isView(value)
  );
}

async function parseResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const errorPayload =
      typeof payload === "object" && payload !== null
        ? (payload as ApiErrorPayload)
        : undefined;

    throw new ApiError(
      response.status,
      errorPayload?.message || response.statusText || "API request failed",
      errorPayload?.details ?? payload,
    );
  }

  return payload as T;
}

export async function request<T>(path: string, options: RequestOptions = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? API_TIMEOUT_MS,
  );

  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort(), {
      once: true,
    });
  }

  try {
    const headers = new Headers(options.headers);
    let body: BodyInit | undefined;

    if (options.body != null) {
      if (isBodyInit(options.body)) {
        body = options.body;
      } else {
        headers.set("Content-Type", "application/json");
        body = JSON.stringify(options.body);
      }
    }

    const response = await fetch(buildUrl(path, options.query), {
      body,
      headers,
      method: options.method ?? "GET",
      signal: controller.signal,
    });

    return parseResponse<T>(response);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(408, "API request timeout");
    }

    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const httpClient = {
  delete: <T>(path: string, options?: Omit<RequestOptions, "method">) =>
    request<T>(path, { ...options, method: "DELETE" }),
  get: <T>(path: string, options?: Omit<RequestOptions, "method" | "body">) =>
    request<T>(path, { ...options, method: "GET" }),
  patch: <T>(path: string, options?: Omit<RequestOptions, "method">) =>
    request<T>(path, { ...options, method: "PATCH" }),
  post: <T>(path: string, options?: Omit<RequestOptions, "method">) =>
    request<T>(path, { ...options, method: "POST" }),
  put: <T>(path: string, options?: Omit<RequestOptions, "method">) =>
    request<T>(path, { ...options, method: "PUT" }),
};
