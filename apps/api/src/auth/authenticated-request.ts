export type AuthenticatedRequest = {
  user: {
    sub: string;
    email: string;
    handle: string;
  };
};
