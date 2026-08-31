import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { buildConnectSrc } from '@/lib/security/csp';
import { shouldRedirectToHttps } from '@/lib/security/transport';

export function middleware(request: NextRequest) {
  // Security Headers
  const nonce = Buffer.from(crypto.randomUUID()).toString('base64');

  // Content Security Policy
  const cspHeader = `
    default-src 'self';
    script-src 'self' 'nonce-${nonce}' 'strict-dynamic' https: ${
      process.env.NODE_ENV === 'production' ? '' : "'unsafe-eval'"
    };
    style-src 'self' 'unsafe-inline';
    img-src 'self' blob: data: https:;
    font-src 'self' data:;
    connect-src ${buildConnectSrc(process.env.NEXT_PUBLIC_API_URL)};
    frame-ancestors 'none';
    base-uri 'self';
    form-action 'self';
  `
    .replace(/\s{2,}/g, ' ')
    .trim();

  // Next applies nonces while rendering by reading the incoming request CSP.
  // Keeping x-nonce alongside it also gives Server Components one canonical
  // request-local nonce without exposing another client authority surface.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-nonce', nonce);
  requestHeaders.set('Content-Security-Policy', cspHeader);

  let response: NextResponse;

  // Public production traffic must be HTTPS. The only exception is explicit
  // loopback traffic used to qualify the built Next server without pretending
  // a local HTTP process terminates TLS.
  if (
    shouldRedirectToHttps({
      nodeEnv: process.env.NODE_ENV,
      forwardedProto: request.headers.get('x-forwarded-proto'),
      hostname: request.nextUrl.hostname,
    })
  ) {
    const redirectUrl = request.nextUrl.clone();
    redirectUrl.protocol = 'https:';
    response = NextResponse.redirect(redirectUrl, 301);
  } else {
    response = NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }

  response.headers.set('Content-Security-Policy', cspHeader);
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=(self)');

  return response;
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - manifest.json (PWA manifest)
     * - sw.js (service worker)
     * - icons (PWA icons)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|manifest.json|sw.js|icon-).*)',
  ],
};
