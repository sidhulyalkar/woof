export interface AuthenticatedUser {
  sub: string;
  email: string;
  handle: string;
}

export interface AuthenticatedRequest {
  user: AuthenticatedUser;
}
