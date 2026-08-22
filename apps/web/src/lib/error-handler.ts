import { toast } from 'sonner';

export interface ApiError {
  message: string;
  statusCode?: number;
  code?: string;
  field?: string;
}

type ErrorPayload = {
  message?: string;
  error?: string;
  code?: string;
  field?: string;
};

type TransportError = {
  response?: {
    data?: ErrorPayload;
    status?: number;
  };
  request?: unknown;
};

export class AppError extends Error {
  statusCode: number;
  code?: string;
  field?: string;

  constructor(message: string, statusCode = 500, code?: string, field?: string) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.field = field;
    this.name = 'AppError';
  }
}

function asTransportError(error: unknown): TransportError {
  return typeof error === 'object' && error !== null ? (error as TransportError) : {};
}

export function handleApiError(error: unknown): AppError {
  const transportError = asTransportError(error);

  if (transportError.response) {
    const { data, status } = transportError.response;
    return new AppError(
      data?.message ?? data?.error ?? 'An error occurred',
      status ?? 500,
      data?.code,
      data?.field
    );
  }

  if (transportError.request) {
    return new AppError('Network error. Please check your connection.', 0, 'NETWORK_ERROR');
  }

  if (error instanceof Error) {
    return new AppError(error.message, 500);
  }

  return new AppError('An unexpected error occurred', 500);
}

export function showErrorToast(error: unknown, defaultMessage = 'An error occurred') {
  const appError = handleApiError(error);
  const errorMessages: Record<number, string> = {
    400: 'Invalid request. Please check your input.',
    401: 'You need to log in to continue.',
    403: "You don't have permission to do this.",
    404: 'The requested resource was not found.',
    409: 'This action conflicts with existing data.',
    429: 'Too many requests. Please slow down.',
    500: 'Server error. Please try again later.',
    503: 'Service temporarily unavailable.',
  };

  const message =
    appError.statusCode && errorMessages[appError.statusCode]
      ? errorMessages[appError.statusCode]
      : appError.message || defaultMessage;

  toast.error(message, {
    description: appError.code ? `Error code: ${appError.code}` : undefined,
  });

  return appError;
}

export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  options?: {
    successMessage?: string;
    errorMessage?: string;
    showSuccessToast?: boolean;
  }
): Promise<T | null> {
  try {
    const result = await fn();
    if (options?.showSuccessToast && options.successMessage) {
      toast.success(options.successMessage);
    }
    return result;
  } catch (error) {
    showErrorToast(error, options?.errorMessage);
    return null;
  }
}

export function getErrorMessage(error: unknown): string {
  return handleApiError(error).message;
}

export function isNetworkError(error: unknown): boolean {
  return handleApiError(error).code === 'NETWORK_ERROR';
}

export function isAuthError(error: unknown): boolean {
  const appError = handleApiError(error);
  return appError.statusCode === 401 || appError.statusCode === 403;
}
