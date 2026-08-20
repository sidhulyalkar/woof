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
      headers: { "Content-Type": "application/json", ...options?.headers },
      ...options,
    })

    if (!response.ok) {
      throw new APIError(`API request failed: ${response.statusText}`, response.status)
    }

    const data = await response.json()
    return data as T
  } catch (error) {
    if (error instanceof APIError) throw error
    throw new APIError("Network error occurred", 500)
  }
}

export function apiGet<T>(endpoint: string): Promise<T> {
  return apiRequest<T>(endpoint, { method: "GET" })
}

export function apiPost<T>(endpoint: string, body: any): Promise<T> {
  return apiRequest<T>(endpoint, { method: "POST", body: JSON.stringify(body) })
}

export function apiPut<T>(endpoint: string, body: any): Promise<T> {
  return apiRequest<T>(endpoint, { method: "PUT", body: JSON.stringify(body) })
}

export function apiDelete<T>(endpoint: string): Promise<T> {
  return apiRequest<T>(endpoint, { method: "DELETE" })
}
