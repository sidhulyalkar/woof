import { ApiProperty } from '@nestjs/swagger';
import { IsEmail, IsOptional, IsString, IsUUID, MaxLength, MinLength } from 'class-validator';

export class RegisterDto {
  @ApiProperty({ example: 'petlover2024' })
  @IsString()
  @MinLength(3)
  @MaxLength(30)
  handle: string;

  @ApiProperty({ example: 'user@example.com' })
  @IsEmail()
  email: string;

  @ApiProperty({ example: 'SecurePass123!' })
  @IsString()
  @MinLength(8)
  password: string;

  @ApiProperty({ example: 'Dog lover from NYC 🐕', required: false })
  @IsOptional()
  @IsString()
  @MaxLength(500)
  bio?: string;

  @ApiProperty({
    example: '7efc01f2-0f7e-45e1-b923-748d6f727ef0',
    required: false,
    description:
      'Client-generated transaction key. Exact retries recover the same email registration; divergent replays fail closed.',
  })
  @IsOptional()
  @IsUUID()
  registrationKey?: string;
}
