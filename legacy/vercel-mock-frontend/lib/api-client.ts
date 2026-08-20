export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
    this.name = "APIError"
  }
}

export async function apiRequest<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`/api${endpoint}`, {
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new APIError(`API request failed: ${response.statusText}`, response.status)
    }

    const data = await response.json()
    return data as T
  } catch (error) {
    if (error instanceof APIError) {
      throw error
    }
    throw new APIError("Network error occurred", 500)
  }
}

export const api = {
  get: <T>(endpoint: string) =>\
    apiRequest<T>(endpoint, { method: "GET" }),

  post: <T>(endpoint: string, body: any) =>
    apiRequest<T>(endpoint, {\
      method: "POST",
      body: JSON.stringify(body),
    }),

  put: <T>(endpoint: string, body: any) =>
    apiRequest<T>(endpoint, {\
      method: "PUT",
      body: JSON.stringify(body),
    }),

  delete: <T>(endpoint: string) =>\
    apiRequest<T>(endpoint, { method: "DELETE" }),\
}
