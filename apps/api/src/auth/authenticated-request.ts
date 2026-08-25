export interface AuthenticatedUser {
  sub: string;
  email: string;
  handle: string;
  sid: string;
}

export interface AuthenticatedRequest {
  user: AuthenticatedUser;
}
