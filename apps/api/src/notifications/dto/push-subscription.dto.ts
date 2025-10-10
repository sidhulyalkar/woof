import { IsString, IsObject, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export class PushSubscriptionKeys {
  @ApiProperty()
  @IsString()
  p256dh: string;

  @ApiProperty()
  @IsString()
  auth: string;
}

export class PushSubscriptionDto {
  @ApiProperty()
  @IsString()
  endpoint: string;

  @ApiProperty({ required: false })
  expirationTime?: number | null;

  @ApiProperty({ type: PushSubscriptionKeys })
  @ValidateNested()
  @Type(() => PushSubscriptionKeys)
  keys: PushSubscriptionKeys;
}

export class SubscribeDto {
  @ApiProperty()
  @IsString()
  userId: string;

  @ApiProperty({ type: PushSubscriptionDto })
  @ValidateNested()
  @Type(() => PushSubscriptionDto)
  subscription: PushSubscriptionDto;
}

export class SendPushDto {
  @ApiProperty()
  @IsString()
  userId: string;

  @ApiProperty()
  @IsString()
  title: string;

  @ApiProperty()
  @IsString()
  body: string;

  @ApiProperty({ required: false })
  @IsObject()
  data?: Record<string, any>;

  @ApiProperty({ required: false })
  @IsString()
  icon?: string;

  @ApiProperty({ required: false })
  @IsString()
  url?: string;
}
