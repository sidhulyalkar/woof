import { Injectable, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { SessionAuthorityService } from '../session-authority.service';

type JwtPayload = {
  sub?: unknown;
  email?: unknown;
  handle?: unknown;
  sid?: unknown;
};

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    configService: ConfigService,
    private readonly sessionAuthority: SessionAuthorityService
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: configService.getOrThrow<string>('JWT_SECRET'),
    });
  }

  async validate(payload: JwtPayload) {
    if (
      typeof payload.sub !== 'string' ||
      typeof payload.email !== 'string' ||
      typeof payload.handle !== 'string' ||
      typeof payload.sid !== 'string'
    ) {
      throw new UnauthorizedException('Session is unavailable');
    }

    await this.sessionAuthority.assertActive(payload.sid, payload.sub);

    return {
      sub: payload.sub,
      email: payload.email,
      handle: payload.handle,
      sid: payload.sid,
    };
  }
}
