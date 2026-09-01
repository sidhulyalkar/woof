import { Type } from 'class-transformer';
import {
  IsNumber,
  IsOptional,
  IsString,
  Matches,
  MinLength,
  ValidateNested,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class PushSubscriptionKeys {
  @ApiProperty()
  @IsString()
  @MinLength(1)
  p256dh: string;

  @ApiProperty()
  @IsString()
  @MinLength(1)
  auth: string;
}

export class PushSubscriptionDto {
  @ApiProperty()
  @IsString()
  @MinLength(1)
  endpoint: string;

  @ApiProperty({ required: false })
  @IsOptional()
  @IsNumber()
  expirationTime?: number | null;

  @ApiProperty({ type: PushSubscriptionKeys })
  @ValidateNested()
  @Type(() => PushSubscriptionKeys)
  keys: PushSubscriptionKeys;
}

export class SubscribeDto {
  @ApiProperty({ type: PushSubscriptionDto })
  @ValidateNested()
  @Type(() => PushSubscriptionDto)
  subscription: PushSubscriptionDto;
}

export class CurrentPushSubscriptionDto {
  @ApiProperty({ description: 'Base64url SHA-256 fingerprint of the browser Push subscription' })
  @IsString()
  @Matches(/^[A-Za-z0-9_-]{43}$/)
  subscriptionFingerprint: string;
}
