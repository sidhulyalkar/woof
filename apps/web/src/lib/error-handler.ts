import { toast } from "sonner"

export interface ApiError {
  message: string
  statusCode?: number
  code?: string
  field?: string
}

export class AppError extends Error {
  statusCode: number
  code?: string
  field?: string

  constructor(message: string, statusCode = 500, code?: string, field?: string) {
    super(message)
    this.statusCode = statusCode
    this.code = code
    this.field = field
    this.name = "AppError"
  }
}

export function handleApiError(error: any): AppError {
  // Axios error with response
  if (error.response) {
    const { data, status } = error.response
    return new AppError(
      data?.message || data?.error || "An error occurred",
      status,
      data?.code,
      data?.field,
    )
  }

  // Axios error without response (network error)
  if (error.request) {
    return new AppError("Network error. Please check your connection.", 0, "NETWORK_ERROR")
  }

  // Other errors
  if (error instanceof Error) {
    return new AppError(error.message, 500)
  }

  return new AppError("An unexpected error occurred", 500)
}

export function showErrorToast(error: any, defaultMessage = "An error occurred") {
  const appError = handleApiError(error)

  const errorMessages: Record<number, string> = {
    400: "Invalid request. Please check your input.",
    401: "You need to log in to continue.",
    403: "You don't have permission to do this.",
    404: "The requested resource was not found.",
    409: "This action conflicts with existing data.",
    429: "Too many requests. Please slow down.",
    500: "Server error. Please try again later.",
    503: "Service temporarily unavailable.",
  }

  const message =
    appError.statusCode && errorMessages[appError.statusCode]
      ? errorMessages[appError.statusCode]
      : appError.message || defaultMessage

  toast.error(message, {
    description: appError.code ? `Error code: ${appError.code}` : undefined,
  })

  return appError
}

export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  options?: {
    successMessage?: string
    errorMessage?: string
    showSuccessToast?: boolean
  },
): Promise<T | null> {
  try {
    const result = await fn()

    if (options?.showSuccessToast && options?.successMessage) {
      toast.success(options.successMessage)
    }

    return result
  } catch (error) {
    showErrorToast(error, options?.errorMessage)
    return null
  }
}

export function getErrorMessage(error: any): string {
  const appError = handleApiError(error)
  return appError.message
}

export function isNetworkError(error: any): boolean {
  const appError = handleApiError(error)
  return appError.code === "NETWORK_ERROR"
}

export function isAuthError(error: any): boolean {
  const appError = handleApiError(error)
  return appError.statusCode === 401 || appError.statusCode === 403
}
