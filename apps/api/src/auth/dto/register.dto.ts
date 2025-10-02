import { ApiProperty } from '@nestjs/swagger';
import { IsEmail, IsString, MinLength, MaxLength, IsOptional } from 'class-validator';

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
}
